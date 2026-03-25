"""
data/preprocessing.py
─────────────────────
Utilities for:
  1. Training a BPE tokeniser on a corpus of .tex files
  2. Rendering entire PDF datasets to pre-cached image tensors (optional speedup)
  3. Building a vocabulary statistics report
"""

from __future__ import annotations

import os
import glob
import json
import argparse
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 1. Tokeniser training
# ─────────────────────────────────────────────────────────────

def collect_tex_files(root: str | Path) -> list[str]:
    """Recursively collect all .tex files under `root`."""
    pattern = str(Path(root) / "**" / "*.tex")
    files = glob.glob(pattern, recursive=True)
    print(f"[preprocessing] Found {len(files)} .tex files under {root}")
    return files


def train_tokenizer(
    corpus_root: str,
    vocab_size: int = 8192,
    output_path: str = "tokenizer.json",
    min_frequency: int = 2,
):
    """
    Train a byte-level BPE tokeniser on all .tex files under `corpus_root`
    and save it to `output_path`.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tex_files = collect_tex_files(corpus_root)
    if not tex_files:
        raise FileNotFoundError(f"No .tex files found under: {corpus_root}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
    )

    print(f"[preprocessing] Training BPE tokeniser (vocab_size={vocab_size}) …")
    tokenizer.train(tex_files, trainer)
    tokenizer.save(output_path)
    print(f"[preprocessing] Tokeniser saved → {output_path}")
    return tokenizer


def load_tokenizer(path: str):
    """Load a saved tokeniser from disk."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


# ─────────────────────────────────────────────────────────────
# 2. PDF pre-rendering cache
# ─────────────────────────────────────────────────────────────

def prerender_dataset(
    data_dir: str,
    output_dir: str,
    dpi: int = 150,
    image_size: int = 896,
):
    """
    Pre-render all PDF pages to PNG files so that training doesn't need
    to run pdflatex per step.

    Output layout::
        output_dir/
          sample_001_page0.png
          sample_001_page1.png
          ...

    A manifest JSON is also written so the dataset can use cached images.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(data_dir.glob("*.pdf"))
    manifest  = {}   # maps (stem, page_idx) → png path

    for pdf_path in tqdm(pdf_files, desc="Rendering PDFs"):
        doc = fitz.open(str(pdf_path))
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            zoom = dpi / 72.0
            pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Resize to fixed image_size
            img = img.resize((image_size, image_size), Image.LANCZOS)

            out_name = f"{pdf_path.stem}_page{page_idx}.png"
            out_path = output_dir / out_name
            img.save(str(out_path))

            key = f"{pdf_path.stem}::{page_idx}"
            manifest[key] = str(out_path)

        doc.close()

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[preprocessing] Rendered {len(manifest)} pages → {output_dir}")
    print(f"[preprocessing] Manifest → {manifest_path}")
    return manifest


# ─────────────────────────────────────────────────────────────
# 3. Cached-image dataset (uses pre-rendered PNGs)
# ─────────────────────────────────────────────────────────────

class CachedPDFLatexDataset:
    """
    Faster variant of PDFLatexDataset that loads pre-rendered PNG images
    instead of calling PyMuPDF on every __getitem__.

    Use after running prerender_dataset() on your data directory.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer,
        image_transform,
        max_seq_len: int = 4096,
        augment: bool = False,
    ):
        import json
        from data.dataset import split_latex_by_page

        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.split_fn = split_latex_by_page

        manifest_path = self.cache_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Build (png_path, tex_chunk) pairs
        self.samples: list[tuple[str, str]] = []
        data_dir = Path(data_dir)
        for key, png_path in self.manifest.items():
            stem, page_str = key.split("::")
            page_idx = int(page_str)
            tex_path = data_dir / f"{stem}.tex"
            if not tex_path.exists():
                continue
            tex_source = tex_path.read_text(encoding="utf-8", errors="replace")
            chunks = self.split_fn(tex_source)
            chunk_idx = page_idx % len(chunks)
            self.samples.append((png_path, chunks[chunk_idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import torch
        from PIL import Image

        png_path, tex_chunk = self.samples[idx]
        image = Image.open(png_path).convert("RGB")

        if self.augment:
            from data.dataset import PDFLatexDataset
            image = PDFLatexDataset._augment_image(None, image)

        pixel_values = self.image_transform(image)

        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        pad_id = self.tokenizer.token_to_id("<pad>")

        encoding  = self.tokenizer.encode(tex_chunk)
        token_ids = [bos_id] + encoding.ids + [eos_id]

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [eos_id]

        seq_len  = len(token_ids)
        padding  = self.max_seq_len - seq_len
        input_ids      = token_ids + [pad_id] * padding
        attention_mask = [1] * seq_len + [0] * padding
        labels = input_ids[1:] + [-100]
        labels = [l if m == 1 else -100 for l, m in zip(labels, attention_mask)]

        return {
            "pixel_values":   pixel_values.float(),
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
            "tex_chunk":      tex_chunk,
        }


# ─────────────────────────────────────────────────────────────
# 4. Vocabulary statistics
# ─────────────────────────────────────────────────────────────

def vocab_statistics(
    corpus_root: str,
    tokenizer_path: str,
    max_files: int = 500,
):
    """
    Print token length statistics over the corpus.
    Useful for choosing max_seq_len.
    """
    import statistics
    tokenizer = load_tokenizer(tokenizer_path)
    tex_files = collect_tex_files(corpus_root)[:max_files]

    lengths = []
    for path in tqdm(tex_files, desc="Computing token lengths"):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        enc  = tokenizer.encode(text)
        lengths.append(len(enc.ids))

    lengths.sort()
    print(f"\n── Token length statistics ({len(lengths)} files) ──")
    print(f"  min:    {min(lengths)}")
    print(f"  median: {statistics.median(lengths):.0f}")
    print(f"  mean:   {statistics.mean(lengths):.0f}")
    print(f"  p95:    {lengths[int(0.95 * len(lengths))]}")
    print(f"  p99:    {lengths[int(0.99 * len(lengths))]}")
    print(f"  max:    {max(lengths)}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF2LaTeX preprocessing utilities")
    sub = parser.add_subparsers(dest="command")

    # train-tokenizer
    p_tok = sub.add_parser("train-tokenizer", help="Train BPE tokeniser on .tex files")
    p_tok.add_argument("--corpus",  required=True, help="Root dir with .tex files")
    p_tok.add_argument("--vocab",   type=int, default=8192)
    p_tok.add_argument("--output",  default="tokenizer.json")
    p_tok.add_argument("--min-freq", type=int, default=2)

    # prerender
    p_pre = sub.add_parser("prerender", help="Pre-render PDFs to PNG cache")
    p_pre.add_argument("--data",    required=True)
    p_pre.add_argument("--output",  required=True)
    p_pre.add_argument("--dpi",     type=int, default=150)
    p_pre.add_argument("--size",    type=int, default=896)

    # vocab-stats
    p_voc = sub.add_parser("vocab-stats", help="Print token length statistics")
    p_voc.add_argument("--corpus",    required=True)
    p_voc.add_argument("--tokenizer", required=True)
    p_voc.add_argument("--max-files", type=int, default=500)

    args = parser.parse_args()

    if args.command == "train-tokenizer":
        train_tokenizer(args.corpus, args.vocab, args.output, args.min_freq)

    elif args.command == "prerender":
        prerender_dataset(args.data, args.output, args.dpi, args.size)

    elif args.command == "vocab-stats":
        vocab_statistics(args.corpus, args.tokenizer, args.max_files)

    else:
        parser.print_help()
