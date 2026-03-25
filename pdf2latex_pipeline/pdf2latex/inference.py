"""
inference.py
────────────
Run a trained PDF2LaTeX model on one or more PDF files and write
the generated .tex output alongside each input file (or to --output-dir).

Usage:
    python inference.py --checkpoint checkpoints/pretrain/best.pt \
                        --config    configs/train_config.yaml \
                        --tokenizer tokenizer.json \
                        --input     paper.pdf \
                        --output-dir results/

    # Process every PDF in a directory:
    python inference.py --checkpoint checkpoints/finetune/best.pt \
                        --config    configs/finetune_config.yaml \
                        --tokenizer tokenizer.json \
                        --input-dir data/test/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.preprocessing import load_tokenizer
from data.dataset import render_pdf_page, count_pdf_pages, build_image_transform
from models.model import build_model


# ─────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def pdf_to_latex(
    pdf_path: str | Path,
    model,
    tokenizer,
    image_transform,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int = 3072,
    temperature: float  = 0.2,   # low temperature for inference → more deterministic
    top_p: float        = 0.9,
    greedy: bool        = False,
) -> list[str]:
    """
    Convert all pages of a PDF to LaTeX strings.

    Returns a list of strings (one per page).
    """
    n_pages = count_pdf_pages(pdf_path)
    results = []

    for page_idx in range(n_pages):
        img = render_pdf_page(pdf_path, page_idx)
        pixel_values = image_transform(img).unsqueeze(0).to(device)  # 1 × 3 × H × W

        gen_ids = model.generate(
            pixel_values,
            bos_id=bos_id,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            greedy=greedy,
        )  # 1 × T

        ids = gen_ids[0].tolist()
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        ids = [i for i in ids if i not in (bos_id,)]
        latex = tokenizer.decode(ids)
        results.append(latex)

    return results


def assemble_document(page_chunks: list[str]) -> str:
    """
    Stitch page-level LaTeX chunks into a single compilable document.
    Uses the first chunk's preamble if present, otherwise adds a minimal one.
    """
    if not page_chunks:
        return ""

    # Check if first chunk already has a preamble
    if r"\documentclass" in page_chunks[0]:
        # Trust the first page's preamble; inject subsequent pages as new pages
        first = page_chunks[0]
        rest  = r"\newpage".join(page_chunks[1:])
        if r"\end{document}" in first:
            combined = first.replace(r"\end{document}", rest + "\n" + r"\end{document}")
        else:
            combined = first + "\n" + rest + "\n" + r"\end{document}"
    else:
        # Wrap everything in a minimal article document
        body = r"\newpage".join(page_chunks)
        combined = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{amsmath,amssymb,graphicx}" + "\n"
            r"\begin{document}" + "\n"
            + body + "\n"
            r"\end{document}"
        )
    return combined


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PDF → LaTeX inference")
    parser.add_argument("--checkpoint",  required=True,  help="Path to model checkpoint (.pt)")
    parser.add_argument("--config",      required=True,  help="Path to train/finetune YAML config")
    parser.add_argument("--tokenizer",   required=True,  help="Path to tokenizer.json")
    parser.add_argument("--input",       default=None,   help="Single PDF file")
    parser.add_argument("--input-dir",   default=None,   help="Directory of PDFs")
    parser.add_argument("--output-dir",  default="outputs", help="Where to write .tex files")
    parser.add_argument("--max-tokens",  type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--greedy",      action="store_true")
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[inference] Using device: {device}")

    # Load tokeniser
    tokenizer = load_tokenizer(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    # Build model
    model = build_model(cfg, pad_id=pad_id).to(device)
    from utils.checkpoint import CheckpointManager
    CheckpointManager.load(args.checkpoint, model, device=device_str)
    model.eval()

    image_transform = build_image_transform(cfg["data"]["image_size"])

    # ── Collect PDFs ─────────────────────────────────────────────────
    pdf_files: list[Path] = []
    if args.input:
        pdf_files.append(Path(args.input))
    if args.input_dir:
        pdf_files.extend(sorted(Path(args.input_dir).glob("*.pdf")))

    if not pdf_files:
        parser.error("Provide --input or --input-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process ──────────────────────────────────────────────────────
    for pdf_path in pdf_files:
        print(f"[inference] Processing: {pdf_path.name} …", end=" ", flush=True)

        page_chunks = pdf_to_latex(
            pdf_path, model, tokenizer, image_transform, device,
            bos_id, eos_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        document = assemble_document(page_chunks)

        out_path = output_dir / (pdf_path.stem + ".tex")
        out_path.write_text(document, encoding="utf-8")
        print(f"→ {out_path}  ({len(page_chunks)} pages)")

    print(f"\n[inference] Done. Wrote {len(pdf_files)} .tex file(s) to {output_dir}/")


if __name__ == "__main__":
    main()
