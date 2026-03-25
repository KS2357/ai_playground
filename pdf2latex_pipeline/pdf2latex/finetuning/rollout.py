"""
finetuning/rollout.py
─────────────────────
Rollout engine: samples sequences from the current policy (model), decodes
them to strings, queries the reward function, and packages everything into
a RolloutBuffer for PPO / GRPO updates.

Key design decisions:
  • No teacher forcing during rollout – pure autoregressive sampling.
  • Stores log-probs from the BEHAVIOUR policy (π_old) alongside observations
    so that the optimiser can compute importance weights.
  • Reference policy (π_ref) log-probs are also stored for KL regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    Stores one batch of rollouts.

    All tensors are B-indexed (batch item) after padding.
    """
    pixel_values:    torch.Tensor          # B × 3 × H × W
    generated_ids:   torch.Tensor          # B × T  (padded with pad_id)
    attention_mask:  torch.Tensor          # B × T  (1 = real token)
    old_log_probs:   torch.Tensor          # B × (T-1)  behaviour policy
    ref_log_probs:   Optional[torch.Tensor]# B × (T-1)  reference policy
    rewards:         torch.Tensor          # B,   scalar per sample
    advantages:      torch.Tensor          # B,   (computed after collection)
    returns:         torch.Tensor          # B,   (same as rewards for bandit)
    tex_chunks:      list[str]             # reference LaTeX (for similarity reward)
    generated_texts: list[str]             # decoded hypotheses


# ─────────────────────────────────────────────────────────────
# Rollout engine
# ─────────────────────────────────────────────────────────────

class RolloutEngine:
    """
    Collects a batch of rollouts from the current policy.

    Parameters
    ----------
    model:           the PDF2LaTeX model (policy π_θ)
    ref_model:       a frozen copy of the model used as π_ref for KL penalty
    reward_fn:       callable(hypotheses, references) → list[float]
    tokenizer:       to decode generated token ids → strings
    bos_id, eos_id, pad_id: special token ids
    max_new_tokens:  max generation length
    temperature:     sampling temperature
    top_p:           nucleus sampling probability
    num_samples:     how many rollouts per prompt (GRPO: group size G)
    """

    def __init__(
        self,
        model,
        ref_model,
        reward_fn,
        tokenizer,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_samples: int = 1,
    ):
        self.model          = model
        self.ref_model      = ref_model
        self.reward_fn      = reward_fn
        self.tokenizer      = tokenizer
        self.bos_id         = bos_id
        self.eos_id         = eos_id
        self.pad_id         = pad_id
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.num_samples    = num_samples

    @torch.no_grad()
    def collect(self, batch: dict, device: torch.device) -> list[RolloutBuffer]:
        """
        Collect rollouts for one batch of prompts.

        Returns a list of `num_samples` RolloutBuffers (one per sample group).
        When num_samples == 1 (PPO), the list has length 1.
        When num_samples > 1 (GRPO), each prompt generates G candidates.
        """
        pixel_values = batch["pixel_values"].to(device)
        tex_chunks   = batch["tex_chunk"]    # list[str]
        B            = pixel_values.size(0)

        all_buffers: list[RolloutBuffer] = []

        for _ in range(self.num_samples):
            # ── 1. Sample from the current policy ──────────────────────
            generated_ids, old_log_probs = self._rollout(pixel_values)
            # generated_ids: B × T  (includes BOS and EOS)
            # old_log_probs: B × (T-1)

            # ── 2. Decode to strings ────────────────────────────────────
            hypotheses = self._decode(generated_ids)

            # ── 3. Score with reward function ───────────────────────────
            rewards = self.reward_fn(hypotheses=hypotheses, references=tex_chunks)
            reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # ── 4. Reference policy log-probs (for KL penalty) ─────────
            attention_mask = self._build_mask(generated_ids)
            if self.ref_model is not None:
                ref_log_probs = self._ref_log_probs(pixel_values, generated_ids,
                                                     attention_mask)
            else:
                ref_log_probs = None

            # ── 5. Advantages = rewards - baseline (filled in by trainer) ─
            advantages = torch.zeros_like(reward_t)
            returns    = reward_t.clone()

            all_buffers.append(RolloutBuffer(
                pixel_values    = pixel_values,
                generated_ids   = generated_ids,
                attention_mask  = attention_mask,
                old_log_probs   = old_log_probs,
                ref_log_probs   = ref_log_probs,
                rewards         = reward_t,
                advantages      = advantages,
                returns         = returns,
                tex_chunks      = tex_chunks,
                generated_texts = hypotheses,
            ))

        return all_buffers

    # ── Internal helpers ──────────────────────────────────────────────

    def _rollout(self, pixel_values: torch.Tensor):
        """Run one forward pass (with sampling) and return ids + log-probs."""
        # model.rollout_with_logprobs returns grad-free tensors here
        # (we call it with no_grad in collect)
        generated_ids, log_probs = self.model.rollout_with_logprobs(
            pixel_values,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return generated_ids, log_probs

    def _decode(self, generated_ids: torch.Tensor) -> list[str]:
        texts = []
        for ids in generated_ids.tolist():
            # Strip BOS; truncate at EOS
            if self.eos_id in ids:
                ids = ids[:ids.index(self.eos_id)]
            ids = [i for i in ids if i != self.bos_id and i != self.pad_id]
            texts.append(self.tokenizer.decode(ids))
        return texts

    def _build_mask(self, generated_ids: torch.Tensor) -> torch.Tensor:
        mask = (generated_ids != self.pad_id).long()
        return mask

    def _ref_log_probs(self, pixel_values, generated_ids, attention_mask):
        return self.ref_model.compute_log_probs(
            pixel_values, generated_ids, attention_mask
        )
