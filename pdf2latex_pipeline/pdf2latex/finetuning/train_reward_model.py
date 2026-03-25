"""
finetuning/train_reward_model.py
────────────────────────────────
Train a reward model on human preference data.

Data format – a JSONL file where each line is:
    {
        "pdf":      "path/to/page.pdf",
        "chosen":   "...preferred LaTeX...",
        "rejected": "...worse LaTeX..."
    }

The reward model is trained with the Bradley-Terry (pairwise) objective:
    L = -log σ(r_chosen - r_rejected)

After training, the checkpoint path is set in configs/finetune_config.yaml
under reward.reward_model_checkpoint.

Usage:
    python finetuning/train_reward_model.py \
        --config      configs/train_config.yaml \
        --pref-data   data/preferences.jsonl \
        --output-dir  checkpoints/reward_model \
        --epochs      5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_image_transform
from data.preprocessing import load_tokenizer
from data.dataset import render_pdf_page
from models.model import build_model
from models.reward_model import build_reward_model
from utils.checkpoint import CheckpointManager
from utils.logging import Logger


# ─────────────────────────────────────────────────────────────
# Preference Dataset
# ─────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Each sample: (page image, chosen LaTeX, rejected LaTeX).
    """

    def __init__(self, jsonl_path: str, tokenizer, image_transform,
                 max_seq_len: int = 2048, dpi: int = 150):
        self.records = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer      = tokenizer
        self.image_transform = image_transform
        self.max_seq_len    = max_seq_len
        self.dpi            = dpi

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Image
        img = render_pdf_page(rec["pdf"], dpi=self.dpi)
        pixel_values = self.image_transform(img)

        # Tokenise chosen and rejected
        chosen_ids,   chosen_mask   = self._tokenise(rec["chosen"])
        rejected_ids, rejected_mask = self._tokenise(rec["rejected"])

        return {
            "pixel_values":   pixel_values.float(),
            "chosen_ids":     chosen_ids,
            "chosen_mask":    chosen_mask,
            "rejected_ids":   rejected_ids,
            "rejected_mask":  rejected_mask,
        }

    def _tokenise(self, text: str):
        pad_id = self.tokenizer.token_to_id("<pad>")
        enc    = self.tokenizer.encode(text)
        ids    = enc.ids[: self.max_seq_len]
        pad    = self.max_seq_len - len(ids)
        ids_t  = torch.tensor(ids + [pad_id] * pad, dtype=torch.long)
        mask_t = torch.tensor([1]*len(ids) + [0]*pad, dtype=torch.long)
        return ids_t, mask_t


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class RewardModelTrainer:
    def __init__(self, cfg: dict, pref_data_path: str, output_dir: str, epochs: int):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        # Tokeniser
        self.tokenizer = load_tokenizer("tokenizer.json")

        # Build the full model just to get an initialised encoder
        pad_id  = self.tokenizer.token_to_id("<pad>")
        policy  = build_model(cfg, pad_id=pad_id)

        # Build reward model (shares the encoder)
        self.rm = build_reward_model(cfg, policy.encoder).to(self.device)

        # Freeze encoder for the first half of training, then unfreeze
        for p in self.rm.vision_encoder.parameters():
            p.requires_grad_(False)

        # Dataset
        img_transform = build_image_transform(cfg["data"]["image_size"])
        ds = PreferenceDataset(
            pref_data_path, self.tokenizer, img_transform,
            max_seq_len=cfg["data"]["max_seq_len"],
            dpi=cfg["data"]["pdf_dpi"],
        )
        self.loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

        # Optimiser
        self.optim = AdamW(self.rm.parameters(), lr=1e-4, weight_decay=0.01)
        self.ckpt  = CheckpointManager(output_dir)
        self.logger = Logger(use_wandb=cfg["training"].get("use_wandb", False),
                             project=cfg["training"].get("wandb_project", "pdf2latex"),
                             run_name="reward-model")

    def train(self):
        print("[rm] Training reward model …")
        global_step = 0
        for epoch in range(self.epochs):

            # Unfreeze encoder halfway through
            if epoch == self.epochs // 2:
                print("[rm] Unfreezing vision encoder")
                for p in self.rm.vision_encoder.parameters():
                    p.requires_grad_(True)

            for batch in self.loader:
                pixel_values  = batch["pixel_values"].to(self.device)
                chosen_ids    = batch["chosen_ids"].to(self.device)
                chosen_mask   = batch["chosen_mask"].to(self.device)
                rejected_ids  = batch["rejected_ids"].to(self.device)
                rejected_mask = batch["rejected_mask"].to(self.device)

                loss = self.rm.preference_loss(
                    pixel_values,
                    chosen_ids,   rejected_ids,
                    chosen_mask,  rejected_mask,
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.rm.parameters(), 1.0)
                self.optim.step()

                global_step += 1
                if global_step % 10 == 0:
                    print(f"[rm epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")
                    self.logger.log({"rm/loss": loss.item()}, step=global_step)

            self.ckpt.save(self.rm, self.optim, global_step,
                           name=f"reward_model_epoch{epoch+1}.pt")

        self.ckpt.save(self.rm, self.optim, global_step, name="best.pt")
        self.logger.finish()
        print("[rm] Done.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--pref-data",  required=True,
                        help="JSONL file with {pdf, chosen, rejected} records")
    parser.add_argument("--output-dir", default="checkpoints/reward_model")
    parser.add_argument("--epochs",     type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    RewardModelTrainer(cfg, args.pref_data, args.output_dir, args.epochs).train()


if __name__ == "__main__":
    main()
