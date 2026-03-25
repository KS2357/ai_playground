"""
models/decoder.py
─────────────────
Autoregressive transformer decoder that attends to vision encoder outputs
and generates LaTeX tokens one at a time.

Architecture:
  • Causal self-attention  (masked so each token only sees past tokens)
  • Cross-attention        (attends to patch embeddings from the encoder)
  • Feed-forward MLP
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Attention primitives
# ─────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 4096):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask (upper triangle = -inf)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # each: B × H × T × D

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B × H × T × T
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
                                float("-inf"))
        # Apply padding mask (True = ignore)
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    """Decoder cross-attends to encoder patch embeddings."""

    def __init__(self, decoder_dim: int, encoder_dim: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        assert decoder_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(decoder_dim, decoder_dim)
        self.k_proj  = nn.Linear(encoder_dim, decoder_dim)
        self.v_proj  = nn.Linear(encoder_dim, decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,
                enc_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        S = enc.size(1)

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # B × H × T × S
        if enc_padding_mask is not None:
            attn = attn.masked_fill(enc_padding_mask.unsqueeze(1).unsqueeze(2),
                                    float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.proj_drop(self.out_proj(x))


# ─────────────────────────────────────────────────────────────
# Decoder block
# ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, decoder_dim: int, encoder_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 max_seq_len: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.self_attn = CausalSelfAttention(decoder_dim, num_heads, dropout, max_seq_len)

        self.norm2 = nn.LayerNorm(decoder_dim)
        self.cross_attn = CrossAttention(decoder_dim, encoder_dim, num_heads, dropout)

        self.norm3 = nn.LayerNorm(decoder_dim)
        mlp_hidden = int(decoder_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, decoder_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, enc: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                enc_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), tgt_key_padding_mask)
        x = x + self.cross_attn(self.norm2(x), enc, enc_padding_mask)
        x = x + self.mlp(self.norm3(x))
        return x


# ─────────────────────────────────────────────────────────────
# Full Decoder
# ─────────────────────────────────────────────────────────────

class LaTeXDecoder(nn.Module):
    """
    Autoregressive decoder that generates LaTeX token sequences.

    Input:
        input_ids:          B × T   (teacher-forced token indices during training)
        encoder_output:     B × S × encoder_dim
        attention_mask:     B × T   (1 = real token, 0 = pad)

    Output:
        logits:             B × T × vocab_size
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 4096,
        decoder_dim: int = 768,
        encoder_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.decoder_dim = decoder_dim

        self.token_embed = nn.Embedding(vocab_size, decoder_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, decoder_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, encoder_dim, num_heads, mlp_ratio, dropout, max_seq_len)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.lm_head = nn.Linear(decoder_dim, vocab_size, bias=False)

        # Tie embedding weights (reduces parameters, improves generalisation)
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # 1 × T
        x = self.emb_drop(self.token_embed(input_ids) + self.pos_embed(positions))

        # Convert padding mask: True = ignore (masked_fill convention)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)  # B × T

        for block in self.blocks:
            x = block(x, encoder_output, tgt_key_padding_mask=pad_mask)

        x = self.norm(x)
        return self.lm_head(x)   # B × T × vocab_size

    # ── Inference / generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive generation with nucleus (top-p) sampling or greedy decoding.

        Returns: B × T token id tensor (padded to longest sequence).
        """
        B = encoder_output.size(0)
        device = encoder_output.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, encoder_output)   # B × T × V
            next_logits = logits[:, -1, :]                     # B × V

            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = _sample_top_p(next_logits, temperature, top_p)

            next_token = next_token.masked_fill(finished.unsqueeze(-1), eos_id)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated   # B × T


# ─────────────────────────────────────────────────────────────
# Sampling helper
# ─────────────────────────────────────────────────────────────

def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    logits = logits / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    # Remove tokens once cumulative probability exceeds top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(dim=-1, index=next_token)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_decoder(cfg: dict) -> LaTeXDecoder:
    m = cfg["model"]
    return LaTeXDecoder(
        vocab_size=m["vocab_size"],
        max_seq_len=m["max_position_embeddings"],
        decoder_dim=m["decoder_dim"],
        encoder_dim=m["encoder_dim"],
        depth=m["decoder_layers"],
        num_heads=m["decoder_heads"],
        mlp_ratio=m["decoder_mlp_ratio"],
        dropout=m["decoder_dropout"],
    )
