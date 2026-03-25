"""
finetuning/reward_functions.py
──────────────────────────────
Reward signals used during rollout-based fine-tuning.

All reward functions are designed to be:
  • Fast enough to run per rollout step
  • Differentiable where possible (but most are non-differentiable "verifier" rewards)
  • Composable via a HybridReward wrapper

Three reward components are provided:
  1. CompilationReward  – does pdflatex compile?                   (0 or 1)
  2. SimilarityReward   – normalised edit-distance to reference     (0 … 1)
  3. FormatReward       – structural heuristics (braces, envs)      (0 … 1)
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable

from training.metrics import (
    check_compilation,
    normalised_edit_distance,
    format_score,
)


# ─────────────────────────────────────────────────────────────
# Individual reward components
# ─────────────────────────────────────────────────────────────

class CompilationReward:
    """
    Binary reward: +1 if the LaTeX compiles, 0 otherwise.

    Uses subprocess pdflatex; runs in a thread pool for parallelism.
    """

    def __init__(self, compiler: str = "pdflatex", timeout: int = 30,
                 n_workers: int = 4):
        self.compiler = compiler
        self.timeout  = timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)

    def __call__(self, latex_strings: list[str]) -> list[float]:
        futures = [
            self.executor.submit(check_compilation, s, self.compiler, self.timeout)
            for s in latex_strings
        ]
        return [float(f.result()) for f in futures]


class SimilarityReward:
    """
    Reward based on similarity to a reference LaTeX string.
    reward = 1 - normalised_edit_distance(hyp, ref)
    """

    def __call__(self, hypotheses: list[str], references: list[str]) -> list[float]:
        return [
            1.0 - normalised_edit_distance(h, r)
            for h, r in zip(hypotheses, references)
        ]


class FormatReward:
    """
    Reward based on structural heuristics (brace balance, environment matching).
    """

    def __call__(self, latex_strings: list[str]) -> list[float]:
        return [format_score(s) for s in latex_strings]


# ─────────────────────────────────────────────────────────────
# Composite reward
# ─────────────────────────────────────────────────────────────

class HybridReward:
    """
    Weighted combination of multiple reward signals.

    Usage::

        reward = HybridReward(
            compilation_weight=0.5,
            similarity_weight=0.3,
            format_weight=0.2,
            compiler="pdflatex",
        )
        rewards = reward(hypotheses=["\\textbf{hi}", ...],
                         references=["\\textbf{hello}", ...])
    """

    def __init__(
        self,
        compilation_weight: float = 0.5,
        similarity_weight:  float = 0.3,
        format_weight:      float = 0.2,
        compiler:           str   = "pdflatex",
        compiler_timeout:   int   = 30,
        reward_model:       object | None = None,
        reward_model_weight: float = 0.0,
        n_workers:          int   = 4,
    ):
        assert abs(compilation_weight + similarity_weight + format_weight
                   + reward_model_weight - 1.0) < 1e-6, \
            "Reward weights must sum to 1.0"

        self.compilation_w    = compilation_weight
        self.similarity_w     = similarity_weight
        self.format_w         = format_weight
        self.rm_w             = reward_model_weight

        self.compilation_fn = CompilationReward(compiler, compiler_timeout, n_workers)
        self.similarity_fn  = SimilarityReward()
        self.format_fn      = FormatReward()
        self.reward_model   = reward_model   # optional learned reward model

    def __call__(
        self,
        hypotheses:  list[str],
        references:  list[str] | None = None,
        pixel_values = None,         # needed for learned reward model
        tokenizer    = None,
    ) -> list[float]:
        """
        Compute per-sample scalar rewards.

        hypotheses:  list of generated LaTeX strings (B,)
        references:  list of ground-truth LaTeX strings (B,)  [may be None]
        """
        B = len(hypotheses)
        rewards = [0.0] * B

        # Compilation
        comp = self.compilation_fn(hypotheses)
        for i, r in enumerate(comp):
            rewards[i] += self.compilation_w * r

        # Similarity (needs references)
        if references and self.similarity_w > 0:
            sim = self.similarity_fn(hypotheses, references)
            for i, r in enumerate(sim):
                rewards[i] += self.similarity_w * r

        # Format
        fmt = self.format_fn(hypotheses)
        for i, r in enumerate(fmt):
            rewards[i] += self.format_w * r

        # Learned reward model
        if self.reward_model is not None and pixel_values is not None and self.rm_w > 0:
            rm_rewards = self._rm_reward(hypotheses, pixel_values, tokenizer)
            for i, r in enumerate(rm_rewards):
                rewards[i] += self.rm_w * r

        return rewards

    def _rm_reward(self, hypotheses, pixel_values, tokenizer) -> list[float]:
        import torch
        device = next(self.reward_model.parameters()).device
        enc = tokenizer.encode_batch(hypotheses)
        ids  = torch.tensor([e.ids for e in enc], dtype=torch.long, device=device)
        mask = torch.tensor([[1 if t != 0 else 0 for t in e.ids] for e in enc],
                            dtype=torch.long, device=device)
        with torch.no_grad():
            scores = self.reward_model(pixel_values.to(device), ids, mask)
        # Normalise to [0, 1] with a sigmoid
        return torch.sigmoid(scores).cpu().tolist()


def build_reward(cfg: dict, reward_model=None) -> HybridReward:
    rc = cfg["reward"]
    rw = rc.get("reward_model_weight", 0.0) if reward_model else 0.0
    # Re-normalise if reward model is disabled
    total = rc["compilation_weight"] + rc["similarity_weight"] + rc["format_weight"]
    return HybridReward(
        compilation_weight=rc["compilation_weight"] / total,
        similarity_weight =rc["similarity_weight"]  / total,
        format_weight     =rc["format_weight"]      / total,
        compiler          =rc.get("latex_compiler", "pdflatex"),
        compiler_timeout  =rc.get("compiler_timeout", 30),
        reward_model      =reward_model,
        reward_model_weight=rw,
    )
