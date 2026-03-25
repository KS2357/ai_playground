"""
training/trainer.py
───────────────────
Supervised pre-training loop for PDF2LaTeX.

Usage:
    python training/trainer.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import yaml

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_dataloaders, build_latex_tokenizer
from models.model import build_model
from training.metrics import token_accuracy, bleu_score, batch_format_score
from utils.checkpoint import CheckpointManager
from utils.logging import Logger


# ─────────────────────────────────────────────────────────────
# LR Scheduler factory
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, total_steps: int):
    warmup = cfg["training"]["warmup_steps"]
    kind   = cfg["training"]["lr_scheduler"]

    warmup_sched = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                            total_iters=warmup)
    if kind == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=1e-7)
    elif kind == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                              total_iters=total_steps - warmup)
    else:  # constant
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=1.0,
                              total_iters=total_steps - warmup)

    return SequentialLR(optimizer, schedulers=[warmup_sched, main_sched],
                        milestones=[warmup])


# ─────────────────────────────────────────────────────────────
# Trainer class
# ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Tokeniser ──────────────────────────────────────────────────
        print("[trainer] Building tokeniser …")
        self.tokenizer = build_latex_tokenizer(
            vocab_size=cfg["model"]["vocab_size"]
        )
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        # ── Data ────────────────────────────────────────────────────────
        print("[trainer] Building dataloaders …")
        self.train_loader, self.val_loader = build_dataloaders(cfg, self.tokenizer)

        steps_per_epoch = len(self.train_loader) // cfg["training"]["gradient_accumulation_steps"]
        self.total_steps = steps_per_epoch * cfg["training"]["epochs"]

        # ── Model ───────────────────────────────────────────────────────
        print("[trainer] Building model …")
        self.model = build_model(cfg, pad_id=self.pad_id).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[trainer] Model params: {n_params:.1f} M")

        # ── Optimiser & scheduler ───────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            betas=(0.9, 0.95),
        )
        self.scheduler = build_scheduler(self.optimizer, cfg, self.total_steps)

        # ── Mixed precision ─────────────────────────────────────────────
        self.use_amp  = cfg["training"].get("fp16", False) or cfg["training"].get("bf16", False)
        self.amp_dtype = torch.bfloat16 if cfg["training"].get("bf16", False) else torch.float16
        self.scaler   = GradScaler(enabled=cfg["training"].get("fp16", False))

        # ── Utilities ───────────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=cfg["training"]["checkpoint_dir"],
            keep_n=cfg["training"]["keep_last_n_checkpoints"],
        )
        self.logger = Logger(
            use_wandb=cfg["training"].get("use_wandb", False),
            project=cfg["training"].get("wandb_project", "pdf2latex"),
            run_name=cfg["training"].get("wandb_run_name", "pretrain"),
            config=cfg,
        )

        self.grad_accum = cfg["training"]["gradient_accumulation_steps"]
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ── Training ───────────────────────────────────────────────────────

    def train(self):
        print(f"[trainer] Starting training for {self.cfg['training']['epochs']} epochs …")
        for epoch in range(self.cfg["training"]["epochs"]):
            self._train_epoch(epoch)

            if (epoch + 1) % max(1, self.cfg["training"]["epochs"] // 10) == 0:
                val_loss = self._validate()
                print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.ckpt_manager.save(self.model, self.optimizer, self.global_step,
                                           name="best.pt")

        self.logger.finish()
        print("[trainer] Training complete.")

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            pixel_values   = batch["pixel_values"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                loss, logits = self.model(pixel_values, input_ids, attention_mask, labels)
                loss = loss / self.grad_accum

            if self.cfg["training"].get("fp16", False):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                if self.cfg["training"].get("fp16", False):
                    self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg["training"]["max_grad_norm"]
                )

                if self.cfg["training"].get("fp16", False):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.cfg["training"]["log_every_n_steps"] == 0:
                    acc = token_accuracy(logits.detach(), labels)
                    lr  = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    self.logger.log({
                        "train/loss":     loss.item() * self.grad_accum,
                        "train/accuracy": acc,
                        "train/lr":       lr,
                        "train/step_time_ms": elapsed * 1000 / max(self.global_step, 1),
                    }, step=self.global_step)

                if (self.global_step % self.cfg["training"]["save_every_n_steps"] == 0):
                    self.ckpt_manager.save(self.model, self.optimizer, self.global_step)

                if (self.global_step % self.cfg["training"]["val_every_n_steps"] == 0):
                    val_loss = self._validate()
                    self.logger.log({"val/loss": val_loss}, step=self.global_step)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.ckpt_manager.save(self.model, self.optimizer,
                                               self.global_step, name="best.pt")
                    self.model.train()

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        max_batches = self.cfg["training"].get("val_max_batches", 999999)
        hyps, refs  = [], []

        for batch in self.val_loader:
            if n_batches >= max_batches:
                break
            pixel_values   = batch["pixel_values"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                loss, _ = self.model(pixel_values, input_ids, attention_mask, labels)
            total_loss += loss.item()
            n_batches  += 1

            # Generate a few samples for qualitative inspection (first batch only)
            if n_batches == 1:
                gen_ids = self.model.generate(
                    pixel_values[:2],
                    bos_id=self.bos_id, eos_id=self.eos_id,
                    max_new_tokens=256, greedy=True,
                )
                for ids in gen_ids:
                    tokens = ids.tolist()
                    if self.eos_id in tokens:
                        tokens = tokens[:tokens.index(self.eos_id)]
                    hyps.append(self.tokenizer.decode(tokens[1:]))  # strip BOS
                for chunk in batch["tex_chunk"][:2]:
                    refs.append(chunk)

        avg_loss = total_loss / max(n_batches, 1)
        if hyps:
            b = bleu_score(hyps, refs)
            fmt = batch_format_score(hyps)
            print(f"  [val] loss={avg_loss:.4f} | BLEU={b:.2f} | format={fmt:.2f}")
            self.logger.log({"val/bleu": b, "val/format_score": fmt},
                            step=self.global_step)

        return avg_loss


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
