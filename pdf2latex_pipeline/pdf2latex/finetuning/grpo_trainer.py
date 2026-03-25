"""
finetuning/grpo_trainer.py
──────────────────────────
GRPO (Group Relative Policy Optimisation) fine-tuning for PDF2LaTeX.

GRPO is the method used in DeepSeek-R1.  Key differences from PPO:
  • No critic / value function needed → simpler, fewer parameters.
  • For each prompt, sample G candidate sequences (the "group").
  • Advantages are computed *relative within the group*:
        A_i = (R_i - mean(R_group)) / std(R_group)
  • Uses the same clipped surrogate objective as PPO, plus KL penalty.

Update objective per group:
    L_GRPO = -E_i[ min(r_i A_i, clip(r_i, 1±ε) A_i) ] + β·KL(π_θ || π_ref)

where r_i = π_θ(y_i | x) / π_old(y_i | x).

Usage:
    python finetuning/grpo_trainer.py --config configs/finetune_config.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_dataloaders, build_latex_tokenizer
from models.model import build_model
from finetuning.rollout import RolloutEngine, RolloutBuffer
from finetuning.reward_functions import build_reward
from utils.checkpoint import CheckpointManager
from utils.logging import Logger


class GRPOTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Tokeniser ─────────────────────────────────────────────────
        self.tokenizer = build_latex_tokenizer(vocab_size=cfg["model"]["vocab_size"])
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        # ── Data ──────────────────────────────────────────────────────
        self.train_loader, self.val_loader = build_dataloaders(cfg, self.tokenizer)

        # ── Policy model ──────────────────────────────────────────────
        self.model = build_model(cfg, pad_id=self.pad_id).to(self.device)
        if cfg["model"].get("checkpoint"):
            self.model.load(cfg["model"]["checkpoint"])
            print(f"[grpo] Loaded policy from {cfg['model']['checkpoint']}")

        # Frozen reference policy π_ref (for KL penalty)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Frozen behaviour policy π_old (refreshed each outer step)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        for p in self.old_model.parameters():
            p.requires_grad_(False)

        # ── Reward ────────────────────────────────────────────────────
        self.reward_fn = build_reward(cfg)

        # ── Rollout engine  ───────────────────────────────────────────
        rc       = cfg["rollout"]
        self.G   = rc["num_samples_per_prompt"]   # group size
        self.rollout_engine = RolloutEngine(
            model          = self.old_model,
            ref_model      = self.ref_model,
            reward_fn      = self.reward_fn,
            tokenizer      = self.tokenizer,
            bos_id         = self.bos_id,
            eos_id         = self.eos_id,
            pad_id         = self.pad_id,
            max_new_tokens = rc["max_new_tokens"],
            temperature    = rc["temperature"],
            top_p          = rc["top_p"],
            num_samples    = self.G,
        )

        # ── Optimiser ─────────────────────────────────────────────────
        tc = cfg["training"]
        self.optim = AdamW(
            self.model.parameters(),
            lr=tc["policy_lr"],
            weight_decay=tc["weight_decay"],
        )
        grpo_cfg = cfg["grpo"]
        self.clip_eps  = grpo_cfg["clip_epsilon"]
        self.kl_coef   = grpo_cfg["kl_coef"]
        self.norm_adv  = grpo_cfg["normalize_advantages"]
        self.max_grad  = tc["max_grad_norm"]

        # ── Utilities ─────────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(cfg["training"]["checkpoint_dir"])
        self.logger = Logger(
            use_wandb=cfg["training"].get("use_wandb", False),
            project=cfg["training"].get("wandb_project", "pdf2latex"),
            run_name=cfg["training"].get("wandb_run_name", "grpo"),
            config=cfg,
        )
        self.global_step  = 0
        self.total_steps  = cfg["training"]["total_steps"]

    # ── Main loop ─────────────────────────────────────────────────────

    def train(self):
        print(f"[grpo] Starting GRPO fine-tuning (G={self.G}) …")
        data_iter = iter(self.train_loader)

        for step in range(self.total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            t0 = time.time()

            # ── 1. Collect G rollouts per prompt ──────────────────────
            buffers = self.rollout_engine.collect(batch, self.device)
            # buffers: list of G RolloutBuffers, each of shape B

            # ── 2. Compute group-relative advantages ──────────────────
            buffers = self._compute_group_advantages(buffers)

            # ── 3. GRPO update ────────────────────────────────────────
            metrics = self._grpo_update(buffers)

            # Sync old model
            self.old_model.load_state_dict(self.model.state_dict())

            metrics["train/mean_reward"]    = torch.stack(
                [b.rewards for b in buffers]).mean().item()
            metrics["train/rollout_time_s"] = time.time() - t0
            self.global_step += 1

            if self.global_step % self.cfg["training"]["log_every_n_steps"] == 0:
                self.logger.log(metrics, step=self.global_step)
                print(f"[grpo step {self.global_step}] "
                      f"reward={metrics['train/mean_reward']:.3f} "
                      f"loss={metrics.get('grpo/loss', 0):.4f} "
                      f"kl={metrics.get('grpo/kl', 0):.5f}")

            if self.global_step % self.cfg["training"]["save_every_n_steps"] == 0:
                self.ckpt_manager.save(self.model, self.optim, self.global_step)

        self.logger.finish()
        print("[grpo] Fine-tuning complete.")

    # ── Group-relative advantage computation ──────────────────────────

    def _compute_group_advantages(self, buffers: list[RolloutBuffer]) -> list[RolloutBuffer]:
        """
        For each prompt i, the G rewards form a group.
        A_{i,g} = (R_{i,g} - mean_g(R_{i,g})) / (std_g(R_{i,g}) + ε)

        Here B prompts × G samples → reward matrix (B × G).
        """
        # Stack rewards: G × B
        reward_stack = torch.stack([b.rewards for b in buffers], dim=0)  # G × B

        group_mean = reward_stack.mean(0, keepdim=True)   # 1 × B
        group_std  = reward_stack.std(0, keepdim=True).clamp(min=1e-8)

        for g, buf in enumerate(buffers):
            adv = (buf.rewards - group_mean.squeeze(0)) / group_std.squeeze(0)
            buf.advantages = adv
        return buffers

    # ── GRPO policy update ────────────────────────────────────────────

    def _grpo_update(self, buffers: list[RolloutBuffer]) -> dict:
        total_loss = 0.0
        total_kl   = 0.0
        total_surr = 0.0

        self.optim.zero_grad()

        for buf in buffers:
            # Recompute log-probs under π_θ
            new_lp = self.model.compute_log_probs(
                buf.pixel_values, buf.generated_ids, buf.attention_mask
            )   # B × (T-1)

            mask = buf.attention_mask[:, 1:].float()
            seq_new_lp = (new_lp * mask).sum(-1)                    # B
            seq_old_lp = (buf.old_log_probs * mask).sum(-1).detach()# B

            ratio = torch.exp(seq_new_lp - seq_old_lp)
            adv   = buf.advantages.detach()

            # Clipped surrogate (same form as PPO)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            surr  = -torch.min(surr1, surr2).mean()

            # KL penalty: KL(π_θ || π_ref)  ≈ Σ [ log π_θ - log π_ref ]
            if buf.ref_log_probs is not None:
                seq_ref_lp = (buf.ref_log_probs * mask).sum(-1).detach()
                kl = (seq_new_lp - seq_ref_lp).mean()
            else:
                kl = torch.tensor(0.0, device=self.device)

            loss = surr + self.kl_coef * kl
            loss = loss / len(buffers)   # average over G samples
            loss.backward()

            total_loss += loss.item()
            total_kl   += kl.item()
            total_surr += surr.item()

        # Gradient step
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
        self.optim.step()

        return {
            "grpo/loss":      total_loss,
            "grpo/surrogate": total_surr / len(buffers),
            "grpo/kl":        total_kl   / len(buffers),
        }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    GRPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
