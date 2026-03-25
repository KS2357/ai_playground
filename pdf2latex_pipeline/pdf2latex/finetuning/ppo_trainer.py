"""
finetuning/ppo_trainer.py
─────────────────────────
PPO (Proximal Policy Optimisation) fine-tuning for PDF2LaTeX.

Algorithm sketch per update step:
  1. Collect K rollouts from π_old (frozen snapshot of current policy)
  2. Compute rewards R and GAE advantages A
  3. For E PPO epochs over the collected data:
       a. Recompute log-probs under π_θ
       b. Compute importance ratio r_t = π_θ / π_old
       c. Clipped surrogate loss:  L_CLIP = E[min(r_t A, clip(r_t, 1±ε) A)]
       d. Value (critic) loss:     L_V    = E[(V_θ(s) - R)²]
       e. KL penalty vs. reference policy π_ref
       f. Entropy bonus
       g. Total: L = -L_CLIP + c1·L_V - c2·H + c3·KL

Usage:
    python finetuning/ppo_trainer.py --config configs/finetune_config.yaml
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
from models.model import PDF2LaTeX, build_model
from finetuning.rollout import RolloutEngine, RolloutBuffer
from finetuning.reward_functions import build_reward
from utils.checkpoint import CheckpointManager
from utils.logging import Logger


# ─────────────────────────────────────────────────────────────
# Value (critic) head
# ─────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Scalar value estimator appended to the policy model.
    Reads the mean of decoder hidden states → scalar V(s).
    We attach it to the encoder output for simplicity (state = image embedding).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """encoder_output: B × S × D  →  B (scalar)."""
        return self.mlp(encoder_output.mean(1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────

class PPOTrainer:
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
            print(f"[ppo] Loaded policy from {cfg['model']['checkpoint']}")

        # Value head
        enc_dim = self.model.encoder.embed_dim
        self.value_head = ValueHead(enc_dim).to(self.device)

        # Frozen reference policy π_ref (for KL penalty)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Frozen behaviour policy π_old (updated each outer step)
        self.old_model = copy.deepcopy(self.model).to(self.device)
        for p in self.old_model.parameters():
            p.requires_grad_(False)

        # ── Reward ────────────────────────────────────────────────────
        self.reward_fn = build_reward(cfg)

        # ── Rollout engine ────────────────────────────────────────────
        rc = cfg["rollout"]
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
            num_samples    = 1,  # PPO: 1 sample per prompt
        )

        # ── Optimisers ────────────────────────────────────────────────
        ppo_cfg = cfg["ppo"]
        tc      = cfg["training"]
        self.policy_optim = AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=tc["policy_lr"], weight_decay=tc["weight_decay"]
        )
        self.clip_eps    = ppo_cfg["clip_epsilon"]
        self.value_coef  = ppo_cfg["value_loss_coef"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.kl_coef     = ppo_cfg["kl_coef"]
        self.target_kl   = ppo_cfg["target_kl"]
        self.ppo_epochs  = ppo_cfg["epochs"]
        self.max_grad    = tc["max_grad_norm"]

        # ── Utilities ─────────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(cfg["training"]["checkpoint_dir"])
        self.logger = Logger(
            use_wandb=cfg["training"].get("use_wandb", False),
            project=cfg["training"].get("wandb_project", "pdf2latex"),
            run_name=cfg["training"].get("wandb_run_name", "ppo"),
            config=cfg,
        )
        self.global_step = 0
        self.total_steps = cfg["training"]["total_steps"]

    # ── Main loop ─────────────────────────────────────────────────────

    def train(self):
        print("[ppo] Starting PPO fine-tuning …")
        data_iter = iter(self.train_loader)

        for step in range(self.total_steps):
            # Get next batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # ── Collect rollouts ──────────────────────────────────────
            t0 = time.time()
            buffers = self.rollout_engine.collect(batch, self.device)
            buf = buffers[0]  # single sample per prompt for PPO

            # Compute advantages
            buf = self._compute_advantages(buf)

            # ── PPO update epochs ──────────────────────────────────────
            metrics = self._ppo_update(buf)
            metrics["train/rollout_time_s"] = time.time() - t0
            metrics["train/mean_reward"]    = buf.rewards.mean().item()
            metrics["train/mean_advantage"] = buf.advantages.mean().item()

            self.global_step += 1

            # Sync old model with current policy
            self.old_model.load_state_dict(self.model.state_dict())

            if self.global_step % self.cfg["training"]["log_every_n_steps"] == 0:
                self.logger.log(metrics, step=self.global_step)
                print(f"[ppo step {self.global_step}] "
                      f"reward={metrics['train/mean_reward']:.3f} "
                      f"policy_loss={metrics.get('ppo/policy_loss', 0):.4f}")

            if self.global_step % self.cfg["training"]["save_every_n_steps"] == 0:
                self.ckpt_manager.save(self.model, self.policy_optim, self.global_step)

        self.logger.finish()
        print("[ppo] Fine-tuning complete.")

    # ── Advantage computation (GAE) ────────────────────────────────────

    def _compute_advantages(self, buf: RolloutBuffer) -> RolloutBuffer:
        """
        For the bandit setting (single-step reward, no value bootstrapping)
        GAE collapses to A = R - V(s).
        """
        with torch.no_grad():
            enc_out = self.model.encoder(buf.pixel_values)
            values  = self.value_head(enc_out)   # B

        advantages = buf.rewards - values
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buf.advantages = advantages
        buf.returns    = buf.rewards
        return buf

    # ── PPO update ──────────────────────────────────────────────────────

    def _ppo_update(self, buf: RolloutBuffer) -> dict:
        metrics: dict = {}
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_kl          = 0.0

        for epoch in range(self.ppo_epochs):
            # Recompute log-probs under current policy
            new_log_probs = self.model.compute_log_probs(
                buf.pixel_values,
                buf.generated_ids,
                buf.attention_mask,
            )   # B × (T-1)

            # Sum token-level log-probs → sequence log-prob
            mask = buf.attention_mask[:, 1:].float()  # B × (T-1)
            seq_new_lp  = (new_log_probs  * mask).sum(-1)   # B
            seq_old_lp  = (buf.old_log_probs * mask).sum(-1).detach()

            ratio = torch.exp(seq_new_lp - seq_old_lp)      # B
            adv   = buf.advantages.detach()

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            enc_out = self.model.encoder(buf.pixel_values)
            values  = self.value_head(enc_out)
            value_loss = F.mse_loss(values, buf.returns.detach())

            # KL against reference policy
            if buf.ref_log_probs is not None:
                seq_ref_lp = (buf.ref_log_probs * mask).sum(-1).detach()
                kl = (seq_new_lp - seq_ref_lp).mean()
            else:
                kl = torch.tensor(0.0, device=self.device)

            # Entropy (approximated as mean negative log-prob)
            entropy = -(new_log_probs * mask).sum(-1).mean()

            # Total loss
            loss = (policy_loss
                    + self.value_coef  * value_loss
                    + self.kl_coef     * kl
                    - self.entropy_coef * entropy)

            self.policy_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.max_grad
            )
            self.policy_optim.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_kl          += kl.item()

            # Early stopping if KL diverges
            if kl.item() > 2 * self.target_kl:
                break

        n = epoch + 1
        metrics["ppo/policy_loss"] = total_policy_loss / n
        metrics["ppo/value_loss"]  = total_value_loss  / n
        metrics["ppo/kl"]          = total_kl           / n
        return metrics


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    PPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
