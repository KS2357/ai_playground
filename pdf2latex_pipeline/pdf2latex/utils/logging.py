"""
utils/logging.py
────────────────
Lightweight logger that writes to stdout and optionally to Weights & Biases.
"""

from __future__ import annotations


class Logger:
    def __init__(
        self,
        use_wandb:  bool = False,
        project:    str  = "pdf2latex",
        run_name:   str  = "run",
        config:     dict | None = None,
    ):
        self.use_wandb = use_wandb
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=project, name=run_name, config=config or {})
                print(f"[logger] W&B run: {wandb.run.url}")
            except ImportError:
                print("[logger] wandb not installed – logging to stdout only.")
                self.use_wandb = False

    def log(self, metrics: dict, step: int | None = None):
        if self._wandb and self.use_wandb:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self._wandb and self.use_wandb:
            self._wandb.finish()
