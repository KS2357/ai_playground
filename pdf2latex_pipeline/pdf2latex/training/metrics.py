"""
training/metrics.py
───────────────────
Evaluation metrics for LaTeX generation.

  • token_accuracy    – exact token match (ignoring pad/eos)
  • bleu_score        – BLEU-4 on tokenised LaTeX strings
  • edit_distance     – normalised character-level edit distance
  • compilation_rate  – fraction of samples that compile with pdflatex
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Sequence

import torch


# ─────────────────────────────────────────────────────────────
# Token accuracy
# ─────────────────────────────────────────────────────────────

def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of non-ignored positions where argmax(logits) == label.

    labels: -100 at positions to ignore.
    """
    preds = logits.argmax(dim=-1)            # B × T
    mask  = labels != -100
    correct = ((preds == labels) & mask).sum().item()
    total   = mask.sum().item()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────

def bleu_score(hypotheses: list[str], references: list[str], n: int = 4) -> float:
    """
    Corpus BLEU-n on whitespace-tokenised strings.

    Avoids heavy NLTK dependency by implementing a simple n-gram BLEU.
    """
    import math
    from collections import Counter

    def ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    bp_num, bp_den = 0, 0
    precisions = [0.0] * n
    counts      = [0]   * n
    total       = [0]   * n

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        bp_num += len(hyp_tokens)
        bp_den += len(ref_tokens)

        for k in range(1, n + 1):
            hyp_ng = ngrams(hyp_tokens, k)
            ref_ng = ngrams(ref_tokens, k)
            clipped = {ng: min(cnt, ref_ng[ng]) for ng, cnt in hyp_ng.items()}
            counts[k-1]  += sum(clipped.values())
            total[k-1]   += max(sum(hyp_ng.values()), 1)

    if min(counts) == 0:
        return 0.0

    log_avg = sum(math.log(c / t) for c, t in zip(counts, total)) / n
    bp = math.exp(min(0, 1 - bp_den / max(bp_num, 1)))
    return bp * math.exp(log_avg) * 100.0


# ─────────────────────────────────────────────────────────────
# Edit distance
# ─────────────────────────────────────────────────────────────

def normalised_edit_distance(hyp: str, ref: str) -> float:
    """
    Levenshtein edit distance normalised by max(len(hyp), len(ref)).
    Returns 0.0 (identical) … 1.0 (completely different).
    """
    try:
        import editdistance
        dist = editdistance.eval(hyp, ref)
    except ImportError:
        # Pure-Python fallback (slow for long strings)
        dist = _levenshtein(hyp, ref)
    return dist / max(len(hyp), len(ref), 1)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def batch_edit_distance(hypotheses: list[str], references: list[str]) -> float:
    """Mean normalised edit distance over a batch."""
    dists = [normalised_edit_distance(h, r) for h, r in zip(hypotheses, references)]
    return sum(dists) / max(len(dists), 1)


# ─────────────────────────────────────────────────────────────
# LaTeX compilation check
# ─────────────────────────────────────────────────────────────

def _wrap_latex(body: str) -> str:
    """Wrap a (possibly partial) LaTeX snippet in a minimal compilable document."""
    if r"\begin{document}" in body:
        return body  # Already a full document
    return (
        r"\documentclass{article}" + "\n"
        r"\begin{document}" + "\n"
        + body + "\n"
        r"\end{document}"
    )


def check_compilation(
    latex_str: str,
    compiler: str = "pdflatex",
    timeout: int = 30,
) -> bool:
    """
    Return True if `latex_str` compiles without errors.

    Runs the compiler in a temporary directory; cleans up afterwards.
    Returns False on timeout or any non-zero exit code.
    """
    doc = _wrap_latex(latex_str)
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "doc.tex")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(doc)
        try:
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", tmpdir, tex_file],
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def batch_compilation_rate(
    latex_strings: list[str],
    compiler: str = "pdflatex",
    timeout: int = 30,
) -> float:
    """Fraction of strings that compile successfully."""
    results = [check_compilation(s, compiler, timeout) for s in latex_strings]
    return sum(results) / max(len(results), 1)


# ─────────────────────────────────────────────────────────────
# LaTeX structural checks (fast, no compiler needed)
# ─────────────────────────────────────────────────────────────

def format_score(latex_str: str) -> float:
    """
    Heuristic structural quality score ∈ [0, 1].

    Checks:
      - Balanced braces {}
      - Balanced brackets []
      - Every \\begin{X} has a matching \\end{X}
      - No obviously truncated sequences
    """
    score = 1.0
    penalties = 0

    # Brace balance
    depth = 0
    for ch in latex_str:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            penalties += 1
            depth = 0
    if depth != 0:
        penalties += 1

    # Bracket balance
    depth = 0
    for ch in latex_str:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if depth < 0:
            penalties += 1
            depth = 0

    # Environment matching
    begins = re.findall(r"\\begin\{(\w+)\}", latex_str)
    ends   = re.findall(r"\\end\{(\w+)\}", latex_str)
    if begins != ends:
        penalties += max(abs(len(begins) - len(ends)), 1)

    # Penalise very short outputs (likely truncated)
    if len(latex_str.strip()) < 10:
        penalties += 3

    score = max(0.0, 1.0 - penalties * 0.2)
    return score


def batch_format_score(latex_strings: list[str]) -> float:
    scores = [format_score(s) for s in latex_strings]
    return sum(scores) / max(len(scores), 1)
