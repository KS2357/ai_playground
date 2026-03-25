"""
models/encoder.py
─────────────────
Vision encoder: a ViT-style patch-embedding transformer that converts a
document-page image into a sequence of contextual patch embeddings.

Can be initialised from a pretrained HuggingFace ViT checkpoint or trained
from scratch.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and linearly project."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Conv2d is equivalent to a per-patch linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × C × H × W  →  B × num_patches × embed_dim
        x = self.proj(x)                 # B × D × H/P × W/P
        x = x.flatten(2).transpose(1, 2)  # B × N × D
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # 3 × B × H × N × D
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B × H × N × N
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# Full Vision Encoder
# ─────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    ViT-style vision encoder.

    Input:  pixel_values  B × 3 × H × W
    Output: patch_embeddings  B × num_patches × encoder_dim
    """

    def __init__(
        self,
        image_size: int = 896,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)

        x = self.patch_embed(pixel_values)                       # B × N × D
        cls = self.cls_token.expand(B, -1, -1)                  # B × 1 × D
        x = torch.cat([cls, x], dim=1)                           # B × (N+1) × D
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Return all patch tokens (excluding CLS) as the context for the decoder
        return x[:, 1:, :]   # B × N × D


# ─────────────────────────────────────────────────────────────
# HuggingFace ViT wrapper (pretrained backbone)
# ─────────────────────────────────────────────────────────────

class HFVisionEncoder(nn.Module):
    """
    Thin wrapper around a HuggingFace ViT model so it can serve as a
    drop-in replacement for VisionEncoder.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224",
                 output_dim: int = 768):
        super().__init__()
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained(model_name)
        vit_dim = self.vit.config.hidden_size

        # Project to the desired output dimension if needed
        self.proj = (nn.Linear(vit_dim, output_dim)
                     if vit_dim != output_dim else nn.Identity())
        self.embed_dim = output_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=pixel_values)
        # last_hidden_state: B × (1 + num_patches) × D
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # drop CLS
        return self.proj(patch_tokens)


def build_encoder(cfg: dict) -> nn.Module:
    """Factory: return the right encoder from config."""
    model_cfg = cfg["model"]
    if model_cfg.get("init_from_pretrained") and model_cfg.get("vision_backbone"):
        return HFVisionEncoder(
            model_name=model_cfg["vision_backbone"],
            output_dim=model_cfg["encoder_dim"],
        )
    return VisionEncoder(
        image_size=cfg["data"]["image_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["encoder_dim"],
        depth=model_cfg["encoder_layers"],
        num_heads=model_cfg["encoder_heads"],
        mlp_ratio=model_cfg["encoder_mlp_ratio"],
        dropout=model_cfg["encoder_dropout"],
    )
