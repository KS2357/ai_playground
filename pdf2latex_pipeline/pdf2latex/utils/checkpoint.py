"""
utils/checkpoint.py
───────────────────
Checkpoint saving / loading with automatic rotation (keep last N).
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_n: int = 3):
        self.dir    = Path(checkpoint_dir)
        self.keep_n = keep_n
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer=None, step: int = 0, name: str | None = None):
        fname = name or f"ckpt_step_{step:07d}.pt"
        path  = self.dir / fname
        payload = {"step": step, "model": model.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, str(path))
        print(f"[checkpoint] Saved → {path}")

        # Rotate – keep last N (skip named checkpoints like best.pt)
        if name is None:
            self._rotate()

    def _rotate(self):
        ckpts = sorted(
            glob.glob(str(self.dir / "ckpt_step_*.pt")),
            key=os.path.getmtime,
        )
        for old in ckpts[: -self.keep_n]:
            os.remove(old)
            print(f"[checkpoint] Removed old checkpoint: {old}")

    def load_latest(self, model, optimizer=None, device="cpu") -> int:
        ckpts = sorted(glob.glob(str(self.dir / "ckpt_step_*.pt")),
                       key=os.path.getmtime)
        if not ckpts:
            print("[checkpoint] No checkpoint found, starting from scratch.")
            return 0
        return self.load(ckpts[-1], model, optimizer, device)

    @staticmethod
    def load(path: str, model, optimizer=None, device="cpu") -> int:
        payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["model"])
        if optimizer and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        step = payload.get("step", 0)
        print(f"[checkpoint] Loaded from {path} (step {step})")
        return step
