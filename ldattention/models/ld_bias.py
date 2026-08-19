"""LD-aware additive attention bias.

This module produces an additive attention-bias tensor of shape ``[B, H, L, L]``
that can be dropped into *any* transformer attention:

- ``torch.nn.functional.scaled_dot_product_attention(..., attn_mask=bias)``
- ``torch.nn.MultiheadAttention(..., attn_mask=bias)`` (reshaped to ``[B*H, L, L]``)
- a HuggingFace ``position_bias`` / added-to-scores slot

Design philosophy: the model is LD-*aware*, not an LD *calculator*. It never
computes an explicit LD / ``r^2`` matrix. The LD-like structure is captured by
two fully learned, symmetric terms that mirror how LD behaves:

1. A per-head distance-decay bias over log-spaced genomic-distance buckets
   (T5-style relative bias). LD decays with recombination distance, and that
   decay is roughly log-linear, so log buckets are both expressive and cheap.
2. A per-head symmetric low-rank bilinear term over token embeddings. LD is a
   pairwise correlation between loci, so the data-dependent term is modeled as
   a correlation-like inner product (not a per-token prior).

Everything is pure ``torch`` and runs identically on CPU or GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LDAttentionBias(nn.Module):
    """Learned, symmetric, additive LD-aware attention bias.

    Args:
        hidden_dim: token embedding dimension.
        num_heads: number of attention heads (bias is per-head).
        num_distance_buckets: number of log-spaced genomic-distance buckets.
        max_distance: largest pairwise distance resolved by the buckets, in the
            same units as ``positions`` (default assumes positions normalized to
            ``[0, 1]``; set to e.g. base pairs or centimorgans otherwise).
        min_distance: smallest positive distance boundary (buckets are log-spaced
            between ``min_distance`` and ``max_distance``).
        genotype_rank: rank of the low-rank bilinear genotype-context term.
        use_genotype_bias: enable the data-dependent genotype-context term.
        num_populations: if > 0, enables per-population FiLM gating of the two
            bias terms for cross-cohort conditioning.
        mask_value: finite additive value used for masked keys (finite instead of
            ``-inf`` to stay NaN-safe under fused/flash SDPA kernels).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_distance_buckets: int = 64,
        max_distance: float = 1.0,
        min_distance: float = 1e-4,
        genotype_rank: int = 16,
        use_distance_bias: bool = True,
        use_genotype_bias: bool = True,
        num_populations: int = 0,
        mask_value: float = -1e9,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_distance_buckets < 2:
            raise ValueError("num_distance_buckets must be >= 2")
        if not 0.0 < min_distance < max_distance:
            raise ValueError("require 0 < min_distance < max_distance")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_distance_buckets = num_distance_buckets
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.use_distance_bias = use_distance_bias
        self.use_genotype_bias = use_genotype_bias
        self.num_populations = num_populations
        self.mask_value = float(mask_value)

        if use_distance_bias:
            # Log-spaced bucket boundaries. bucketize maps a distance to one of
            # `num_distance_buckets` bins (below first boundary -> 0, above last
            # -> num_distance_buckets - 1), so we need num_distance_buckets - 1 edges.
            boundaries = torch.logspace(
                math.log10(self.min_distance),
                math.log10(self.max_distance),
                steps=num_distance_buckets - 1,
            )
            self.register_buffer("distance_boundaries", boundaries, persistent=False)

            # Per-head, per-bucket learnable distance bias. Zero-init so training
            # starts from a plain (unbiased) transformer and learns decay.
            self.distance_bias = nn.Embedding(num_distance_buckets, num_heads)
            nn.init.zeros_(self.distance_bias.weight)

        if use_genotype_bias:
            self.genotype_rank = genotype_rank
            self.geno_proj = nn.Linear(hidden_dim, num_heads * genotype_rank)
            self.geno_scale = genotype_rank**-0.5

        if num_populations > 0:
            # Residual gates around 1.0 for the [distance, genotype] terms.
            self.pop_film = nn.Embedding(num_populations, num_heads * 2)
            nn.init.zeros_(self.pop_film.weight)

    def _distance_bias(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, L] -> pairwise |Δpos|: [B, L, L]
        dist = (positions[:, :, None] - positions[:, None, :]).abs()
        bucket_ids = torch.bucketize(dist, self.distance_boundaries.to(dist.device))
        # [B, L, L, H] -> [B, H, L, L]; symmetric because dist is symmetric.
        return self.distance_bias(bucket_ids).permute(0, 3, 1, 2)

    def _genotype_bias(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = token_embeddings.shape
        proj = self.geno_proj(token_embeddings)  # [B, L, H*r]
        proj = proj.view(b, seq_len, self.num_heads, self.genotype_rank)
        # Symmetric low-rank correlation-like term: same projection for i and j.
        bias = torch.einsum("bihr,bjhr->bhij", proj, proj)
        return bias * self.geno_scale

    def forward(
        self,
        positions: torch.Tensor,
        token_embeddings: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the additive LD-aware attention bias.

        Args:
            positions: ``[B, L]`` or ``[B, L, 1]`` genomic coordinates (any unit
                consistent with ``max_distance``).
            token_embeddings: ``[B, L, C]`` token embeddings for the genotype
                term. If ``None``, only the distance term is used.
            key_padding_mask: ``[B, L]`` where 1/True means keep the token. Masked
                keys receive ``mask_value`` so they are ignored by softmax.
            population_id: ``[B]`` long tensor of population indices (requires
                ``num_populations > 0``).

        Returns:
            Additive bias tensor ``[B, H, L, L]`` (float).
        """
        if positions.dim() == 3:
            positions = positions.squeeze(-1)
        if positions.dim() != 2:
            raise ValueError("positions must be [B, L] or [B, L, 1]")

        batch_size, seq_len = positions.shape

        dist_bias = self._distance_bias(positions) if self.use_distance_bias else None
        geno_bias = None
        if self.use_genotype_bias and token_embeddings is not None:
            geno_bias = self._genotype_bias(token_embeddings)

        if self.num_populations > 0 and population_id is not None:
            gates = 1.0 + self.pop_film(population_id)  # [B, 2H], centered at 1
            if dist_bias is not None:
                dist_bias = dist_bias * gates[:, : self.num_heads, None, None]
            if geno_bias is not None:
                geno_bias = geno_bias * gates[:, self.num_heads :, None, None]

        terms = [t for t in (dist_bias, geno_bias) if t is not None]
        if terms:
            bias = terms[0] if len(terms) == 1 else terms[0] + terms[1]
        else:
            # No LD bias (plain-transformer control); zeros are a valid additive
            # bias and keep the masking path below uniform.
            bias = torch.zeros(
                batch_size, self.num_heads, seq_len, seq_len,
                device=positions.device, dtype=torch.float32,
            )

        if key_padding_mask is not None:
            keep = key_padding_mask.to(torch.bool)[:, None, None, :]  # [B,1,1,L]
            bias = bias.masked_fill(~keep, self.mask_value)

        return bias
