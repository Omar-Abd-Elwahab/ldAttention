"""Core LD-aware attention components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldattention.models.ld_bias import LDAttentionBias


class LDAwareSelfAttention(nn.Module):
    """
    Multi-head self-attention with a learned, LD-aware additive bias.

    Instead of injecting an explicit LD matrix, the attention logits are biased
    by :class:`LDAttentionBias`, which contributes:

    1. a per-head distance-decay bias over log-spaced genomic distances, and
    2. a per-head symmetric genotype-context (correlation-like) bias.

    By default the forward pass uses ``F.scaled_dot_product_attention`` so it
    benefits from fused/flash/memory-efficient kernels on GPU. Pass
    ``need_weights=True`` to take a manual path that also returns the attention
    weight matrix (useful for interpretability / validating against known LD).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_distance_buckets: int = 64,
        max_distance: float = 1.0,
        genotype_rank: int = 16,
        use_distance_bias: bool = True,
        use_genotype_bias: bool = True,
        num_populations: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ld_bias = LDAttentionBias(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_distance_buckets=num_distance_buckets,
            max_distance=max_distance,
            genotype_rank=genotype_rank,
            use_distance_bias=use_distance_bias,
            use_genotype_bias=use_genotype_bias,
            num_populations=num_populations,
        )
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, L, D]

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: [B, L, C] token embeddings.
            positions: [B, L, 1] (or [B, L]) genomic positions.
            attention_mask: optional mask [B, L], where 1 means keep token.
            population_id: optional [B] population indices for FiLM conditioning.
            need_weights: if True, return attention weights (manual path).
        Returns:
            updated embeddings and (optionally) attention weights.
        """
        batch_size, seq_len, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        active = self.ld_bias.has_bias_terms or attention_mask is not None
        scale = self.head_dim**-0.5

        if need_weights:
            # Interpretability path: materialize the bias so it can be read out.
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if active:
                attn_logits = attn_logits + self.ld_bias(
                    positions=positions,
                    token_embeddings=x,
                    key_padding_mask=attention_mask,
                    population_id=population_id,
                ).to(q.dtype)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attended = torch.matmul(attn_weights, v)  # [B, H, L, D]
        else:
            # Training/inference path. The genotype term rides along inside the
            # query and key vectors instead of becoming an [B, H, L, L] mask,
            # and the distance term is broadcast from a single matrix, so the
            # quadratic tensor is never built. With no bias at all this reduces
            # to a plain fused attention call.
            mask = z_query = z_key = None
            if active:
                mask, z_query, z_key = self.ld_bias.attention_terms(
                    positions=positions,
                    token_embeddings=x,
                    key_padding_mask=attention_mask,
                    population_id=population_id,
                )
            if z_query is not None:
                q = torch.cat([q * scale, z_query.to(q.dtype)], dim=-1)
                k = torch.cat([k, z_key.to(k.dtype)], dim=-1)
                scale = 1.0  # the 1/sqrt(d) factor is already inside q
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if mask is None else mask.to(q.dtype),
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale,
            )
            attn_weights = None

        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(attended)
        return out, attn_weights
