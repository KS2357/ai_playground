"""
evaluate.py
───────────
Evaluate a trained PDF2LaTeX checkpoint on a held-out test set.

Computes:
  • Token accuracy (teacher-forced)
  • BLEU-4 on decoded outputs vs. reference LaTeX
  • Mean normalised edit distance
  • LaTeX format score (heuristic)
  • Compilation rate  (requires pdflatex / tectonic in PATH)

Usage:
    python evaluate.py \
        --checkpoint checkpoints/pretrain/best.pt \
        --config     configs/train_config.yaml \
        --tokenizer  tokenizer.json \
        --data-dir   data/test \
        --compiler   pdflatex          # or tectonic; set "none" to skip
        --output     eval_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from data.dataset import build_dataloaders, PDFLatexDataset, build_image_transform
from data.preprocessing import load_tokenizer
from models.model import build_model
from training.metrics import (
    token_accuracy,
    bleu_score,
    batch_edit_distance,
    batch_format_score,
    batch_compilation_rate,
)
from utils.checkpoint import CheckpointManager


def evaluate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # ── Tokeniser ────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(cfg, pad_id=pad_id).to(device)
    CheckpointManager.load(args.checkpoint, model, device=str(device))
    model.eval()

    # ── Data ─────────────────────────────────────────────────────────
    img_transform = build_image_transform(cfg["data"]["image_size"])
    ds = PDFLatexDataset(
        root=args.data_dir,
        tokenizer=tokenizer,
        image_transform=img_transform,
        max_seq_len=cfg["data"]["max_seq_len"],
        dpi=cfg["data"]["pdf_dpi"],
        augment=False,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # ── Accumulate metrics ───────────────────────────────────────────
    total_accuracy = 0.0
    hypotheses: list[str] = []
    references:  list[str] = []
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            if args.max_batches and n_batches >= args.max_batches:
                break

            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Teacher-forced accuracy
            _, logits = model(pixel_values, input_ids, attention_mask, labels)
            total_accuracy += token_accuracy(logits, labels)

            # Generate (greedy for speed during eval)
            gen_ids = model.generate(
                pixel_values,
                bos_id=bos_id, eos_id=eos_id,
                max_new_tokens=min(512, cfg["data"]["max_seq_len"]),
                greedy=True,
            )

            for ids in gen_ids.tolist():
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                ids = [i for i in ids if i != bos_id and i != pad_id]
                hypotheses.append(tokenizer.decode(ids))

            for chunk in batch["tex_chunk"]:
                references.append(chunk)

            n_batches += 1
            print(f"  Batch {n_batches} / {len(loader)}", end="\r")

    elapsed = time.time() - t0
    print(f"\n[eval] Inference time: {elapsed:.1f}s  ({len(hypotheses)} samples)")

    # ── Metrics ──────────────────────────────────────────────────────
    results = {
        "token_accuracy":  total_accuracy / max(n_batches, 1),
        "bleu4":           bleu_score(hypotheses, references),
        "edit_distance":   batch_edit_distance(hypotheses, references),
        "format_score":    batch_format_score(hypotheses),
    }

    if args.compiler and args.compiler.lower() != "none":
        print("[eval] Checking compilation rates (this may take a while) …")
        results["compilation_rate"] = batch_compilation_rate(
            hypotheses[:args.max_compile],
            compiler=args.compiler,
            timeout=30,
        )

    print("\n── Evaluation Results ──────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<25} {v:.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\n[eval] Saved → {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--config",       required=True)
    parser.add_argument("--tokenizer",    required=True)
    parser.add_argument("--data-dir",     required=True)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--max-batches",  type=int, default=None)
    parser.add_argument("--max-compile",  type=int, default=50,
                        help="How many samples to check for compilation (slow)")
    parser.add_argument("--compiler",     default="pdflatex",
                        help="LaTeX compiler binary (or 'none' to skip)")
    parser.add_argument("--output",       default="eval_results.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
