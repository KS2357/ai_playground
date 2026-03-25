"""
models/model.py
───────────────
PDF2LaTeX  –  the full encoder-decoder model.

  VisionEncoder  →  patch embeddings  →  LaTeXDecoder  →  token logits

A thin projection layer bridges encoder_dim → decoder_dim when they differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.encoder import VisionEncoder, HFVisionEncoder, build_encoder
from models.decoder import LaTeXDecoder, build_decoder


class PDF2LaTeX(nn.Module):
    """
    End-to-end transformer model: PDF page image → LaTeX token sequence.

    During training:  forward() returns cross-entropy loss (teacher forcing).
    During inference: generate() returns sampled token ids.
    """

    def __init__(self, encoder: nn.Module, decoder: LaTeXDecoder, pad_id: int = 0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

        encoder_dim = getattr(encoder, "embed_dim", None)
        decoder_dim = decoder.decoder_dim

        # Projection in case dims differ
        if encoder_dim is not None and encoder_dim != decoder_dim:
            self.enc_proj = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.enc_proj = nn.Identity()

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    # ── Forward (training) ────────────────────────────────────────────

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, logits).

        pixel_values:   B × 3 × H × W
        input_ids:      B × T  (BOS + token ids; last token is EOS or PAD)
        attention_mask: B × T  (1 = real, 0 = pad)
        labels:         B × T  (shifted input_ids; -100 at padding positions)
        """
        enc_out = self.encoder(pixel_values)          # B × S × enc_dim
        enc_out = self.enc_proj(enc_out)               # B × S × dec_dim

        logits = self.decoder(input_ids, enc_out, attention_mask)  # B × T × V

        # logits: B × T × V,  labels: B × T  → flatten for CE
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return loss, logits

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        greedy: bool = False,
    ) -> torch.Tensor:
        enc_out = self.encoder(pixel_values)
        enc_out = self.enc_proj(enc_out)
        return self.decoder.generate(
            enc_out, bos_id, eos_id, max_new_tokens, temperature, top_p, greedy
        )

    # ── Rollout (with gradient) for PPO / GRPO ─────────────────────────

    def rollout_with_logprobs(
        self,
        pixel_values: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sequence AND return the log-probabilities of each chosen token.

        Returns:
            generated_ids:  B × T
            log_probs:      B × (T-1)   (log-prob of tokens[1:])
        """
        enc_out = self.encoder(pixel_values)
        enc_out = self.enc_proj(enc_out)

        B = enc_out.size(0)
        device = enc_out.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        log_probs_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decoder(generated, enc_out)   # B × T × V
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-p sampling
            from models.decoder import _sample_top_p
            import torch.nn.functional as F
            probs = F.softmax(next_logits, dim=-1)
            next_token = _sample_top_p(next_logits, temperature=1.0, top_p=top_p)

            log_prob = torch.log(probs.gather(1, next_token) + 1e-9)  # B × 1
            log_probs_list.append(log_prob)

            next_token = next_token.masked_fill(finished.unsqueeze(-1), eos_id)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        log_probs = torch.cat(log_probs_list, dim=1)  # B × (T-1)
        return generated, log_probs

    # ── Compute log-probs for already-sampled sequences ────────────────

    def compute_log_probs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given already-generated sequences, return per-token log-probs.
        Used by PPO / GRPO to evaluate current policy on old rollout data.

        Returns: B × (T-1)
        """
        import torch.nn.functional as F
        enc_out = self.encoder(pixel_values)
        enc_out = self.enc_proj(enc_out)
        logits = self.decoder(input_ids, enc_out, attention_mask)  # B × T × V
        log_probs = F.log_softmax(logits, dim=-1)                  # B × T × V
        # Gather log-prob of each generated token (skip last position)
        token_log_probs = log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)   # B × (T-1)
        return token_log_probs

    # ── Checkpoint I/O ────────────────────────────────────────────────

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))

    def load(self, path: str | Path, strict: bool = True):
        state = torch.load(str(path), map_location="cpu")
        self.load_state_dict(state, strict=strict)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, pad_id: int = 0) -> PDF2LaTeX:
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    model = PDF2LaTeX(
        encoder=encoder,
        decoder=decoder,
        pad_id=pad_id,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )

    if cfg["model"].get("init_from_pretrained") and cfg["model"].get("vision_backbone"):
        print(f"[model] Loaded pretrained vision backbone: {cfg['model']['vision_backbone']}")

    return model
