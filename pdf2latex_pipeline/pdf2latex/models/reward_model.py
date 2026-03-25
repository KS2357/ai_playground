"""
models/reward_model.py
──────────────────────
Learned reward model that scores (PDF-page image, LaTeX-candidate) pairs.

Used *optionally* in the fine-tuning stage – you can train it on human
preference data (preferred vs. rejected LaTeX) via a Bradley-Terry loss.

Architecture: the same VisionEncoder + a small text encoder over the LaTeX
candidate, fused and projected to a scalar reward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """
    Scores a (page-image, latex-candidate) pair → scalar reward ∈ ℝ.

    Training objective: Bradley-Terry preference loss
        L = -log σ(r_chosen - r_rejected)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        latex_encoder:  nn.Module,
        encoder_dim:    int = 768,
        hidden_dim:     int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.latex_encoder  = latex_encoder

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """B × 3 × H × W  →  B × D  (mean-pooled patch embeddings)."""
        patch_emb = self.vision_encoder(pixel_values)   # B × N × D
        return patch_emb.mean(dim=1)                     # B × D

    def encode_latex(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """B × T  →  B × D."""
        return self.latex_encoder(input_ids, attention_mask)   # B × D

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns reward scalar per sample: B,"""
        img_emb   = self.encode_image(pixel_values)
        latex_emb = self.encode_latex(input_ids, attention_mask)
        fused = torch.cat([img_emb, latex_emb], dim=-1)
        return self.fusion(fused).squeeze(-1)   # B

    def preference_loss(
        self,
        pixel_values:       torch.Tensor,
        chosen_ids:         torch.Tensor,
        rejected_ids:       torch.Tensor,
        chosen_mask:        Optional[torch.Tensor] = None,
        rejected_mask:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bradley-Terry loss over (chosen, rejected) LaTeX pairs."""
        r_chosen   = self.forward(pixel_values, chosen_ids,   chosen_mask)
        r_rejected = self.forward(pixel_values, rejected_ids, rejected_mask)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(str(path), map_location="cpu"))


# ─────────────────────────────────────────────────────────────
# Small LaTeX text encoder (bi-directional transformer)
# ─────────────────────────────────────────────────────────────

class LaTeXTextEncoder(nn.Module):
    """
    A small bi-directional transformer that encodes a LaTeX string
    into a fixed-size embedding (mean pool over non-padding tokens).
    """

    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 4,
                 num_heads: int = 8, max_seq_len: int = 4096):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # TransformerEncoder uses src_key_padding_mask where True = ignore
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return x   # B × dim


def build_reward_model(cfg: dict, vision_encoder: nn.Module) -> RewardModel:
    vocab_size   = cfg["model"]["vocab_size"]
    encoder_dim  = cfg["model"]["encoder_dim"]

    latex_encoder = LaTeXTextEncoder(
        vocab_size=vocab_size,
        dim=encoder_dim,
        depth=4,
        num_heads=8,
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    return RewardModel(
        vision_encoder=vision_encoder,
        latex_encoder=latex_encoder,
        encoder_dim=encoder_dim,
    )
