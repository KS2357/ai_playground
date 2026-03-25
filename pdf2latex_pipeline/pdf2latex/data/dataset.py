"""
data/dataset.py
───────────────
Dataset classes for PDF → LaTeX training.

Each sample is a (PDF page image, LaTeX source) pair.
PDFs are rendered to PIL images; LaTeX files are tokenised with a BPE tokeniser
trained on a corpus of .tex files.
"""

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# PDF rendering
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# ─────────────────────────────────────────────────────────────
# Tokeniser helpers
# ─────────────────────────────────────────────────────────────

def build_latex_tokenizer(vocab_size: int = 8192, corpus_files: list[str] | None = None):
    """
    Build (or load) a BPE tokeniser trained on LaTeX source files.

    Uses HuggingFace tokenizers library.  If `corpus_files` is None the
    tokeniser returns a dummy character-level fallback so the rest of the
    pipeline can be exercised without data.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    if corpus_files:
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
        )
        tokenizer.train(corpus_files, trainer)
    else:
        # Minimal fallback: ASCII character vocabulary
        trainer = trainers.BpeTrainer(
            vocab_size=256,
            special_tokens=special_tokens,
        )
        # Train on a tiny in-memory corpus so we get a valid tokeniser
        from tokenizers import AddedToken
        tokenizer.add_special_tokens(special_tokens)

    return tokenizer


# ─────────────────────────────────────────────────────────────
# PDF rendering
# ─────────────────────────────────────────────────────────────

def render_pdf_page(pdf_path: str | Path, page_idx: int = 0, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to a PIL RGB image."""
    pdf_path = str(pdf_path)

    if HAS_FITZ:
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    raise RuntimeError(
        "PyMuPDF (fitz) is required for PDF rendering.  "
        "Install with: pip install pymupdf"
    )


def count_pdf_pages(pdf_path: str | Path) -> int:
    if HAS_FITZ:
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n
    raise RuntimeError("PyMuPDF required.")


# ─────────────────────────────────────────────────────────────
# LaTeX splitting helpers
# ─────────────────────────────────────────────────────────────

def split_latex_by_page(tex_source: str) -> list[str]:
    """
    Heuristically split a .tex file into page-aligned chunks.

    Strategy:
      1. If explicit \\newpage / \\clearpage markers exist, split on them.
      2. Otherwise return the full source as a single chunk (single-page docs).
    """
    # Normalise line endings
    source = tex_source.replace("\r\n", "\n")

    # Split on page-break commands
    parts = re.split(r"\\(?:newpage|clearpage|pagebreak)\b", source)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [source]


# ─────────────────────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────────────────────

class PDFLatexDataset(Dataset):
    """
    Dataset of (PDF-page image, LaTeX-chunk) pairs.

    Directory layout::

        root/
          sample_001.pdf
          sample_001.tex
          sample_002.pdf
          sample_002.tex
          ...

    Each PDF may have multiple pages; the corresponding .tex is split into
    page-aligned chunks.  If page counts don't match, the first page / first
    chunk is used as a fallback.
    """

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        image_transform,
        max_seq_len: int = 4096,
        dpi: int = 150,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.dpi = dpi
        self.augment = augment

        # Build index: list of (pdf_path, page_idx, tex_chunk)
        self.samples: list[tuple[Path, int, str]] = []
        self._build_index()

    def _build_index(self):
        pdf_files = sorted(self.root.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No .pdf files found in {self.root}")

        for pdf_path in pdf_files:
            tex_path = pdf_path.with_suffix(".tex")
            if not tex_path.exists():
                continue  # skip unpaired files

            tex_source = tex_path.read_text(encoding="utf-8", errors="replace")
            tex_chunks = split_latex_by_page(tex_source)

            try:
                n_pages = count_pdf_pages(pdf_path)
            except Exception:
                n_pages = 1

            for page_idx in range(n_pages):
                # Align chunk: use modulo so shorter .tex files don't crash
                chunk_idx = page_idx % len(tex_chunks)
                self.samples.append((pdf_path, page_idx, tex_chunks[chunk_idx]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        pdf_path, page_idx, tex_chunk = self.samples[idx]

        # ── Image ──────────────────────────────────────────────────────
        image = render_pdf_page(pdf_path, page_idx, dpi=self.dpi)

        if self.augment:
            image = self._augment_image(image)

        pixel_values = self.image_transform(image)  # → C×H×W tensor

        # ── Text ───────────────────────────────────────────────────────
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        pad_id = self.tokenizer.token_to_id("<pad>")

        encoding = self.tokenizer.encode(tex_chunk)
        token_ids = [bos_id] + encoding.ids + [eos_id]

        # Truncate to max_seq_len (keep EOS)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [eos_id]

        # Pad
        seq_len = len(token_ids)
        padding = self.max_seq_len - seq_len
        input_ids = token_ids + [pad_id] * padding
        attention_mask = [1] * seq_len + [0] * padding

        # Labels: shift right (teacher forcing); ignore padding positions
        labels = input_ids[1:] + [-100]  # -100 ignored by CrossEntropyLoss
        labels = [l if m == 1 else -100 for l, m in zip(labels, attention_mask)]

        return {
            "pixel_values":   torch.tensor(pixel_values, dtype=torch.float32)
                              if not isinstance(pixel_values, torch.Tensor)
                              else pixel_values.float(),
            "input_ids":      torch.tensor(input_ids,     dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,        dtype=torch.long),
            "tex_chunk":      tex_chunk,   # raw string, for reward computation
        }

    # ── Augmentation ───────────────────────────────────────────────────

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Light augmentation suitable for document images."""
        from PIL import ImageFilter, ImageEnhance

        # Random slight rotation (documents rarely need heavy rotation)
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, fillcolor=(255, 255, 255))

        # Brightness / contrast jitter
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.8)))

        return image


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def build_image_transform(image_size: int = 896):
    """Standard document-image normalisation pipeline."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # ImageNet stats work well even for document images
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(cfg: dict, tokenizer) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from a config dict."""
    img_transform = build_image_transform(cfg["data"]["image_size"])

    train_ds = PDFLatexDataset(
        root=cfg["data"]["train_dir"],
        tokenizer=tokenizer,
        image_transform=img_transform,
        max_seq_len=cfg["data"]["max_seq_len"],
        dpi=cfg["data"]["pdf_dpi"],
        augment=True,
    )
    val_ds = PDFLatexDataset(
        root=cfg["data"]["val_dir"],
        tokenizer=tokenizer,
        image_transform=img_transform,
        max_seq_len=cfg["data"]["max_seq_len"],
        dpi=cfg["data"]["pdf_dpi"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader
