"""Imputation task model built on LD-aware encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ldattention.models.encoder import LDAwareEncoder


class FourierPositionEncoding(nn.Module):
    """Sinusoidal features of a *continuous* genomic coordinate.

    A single raw position scalar gives the shared input projection almost no
    ability to tell one variant from another, which caps how site-specific the
    predictions can be. Expanding the coordinate over a geometric ladder of
    frequencies restores that resolution while staying a function of genomic
    position -- so, unlike a per-site lookup table, the encoding still transfers
    to new windows, new variant counts, and new cohorts.
    """

    def __init__(self, num_frequencies: int = 16, max_frequency: float = 256.0) -> None:
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_frequency), steps=num_frequencies) * math.pi
        self.register_buffer("frequencies", freqs, persistent=False)
        self.out_dim = 2 * num_frequencies + 1

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """``[B, L, 1]`` -> ``[B, L, 2F + 1]`` (the raw coordinate is kept)."""
        scaled = positions * self.frequencies
        return torch.cat([positions, scaled.sin(), scaled.cos()], dim=-1)


class LDAwareImputationModel(nn.Module):
    """
    Genotype imputation model.

    Input channels follow your current convention:
    - allele_1
    - allele_2
    Output classes:
    - 0 -> 0/0
    - 1 -> 0/1
    - 2 -> 1/1
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
        num_distance_buckets: int = 64,
        max_distance: float = 1.0,
        genotype_rank: int = 16,
        use_distance_bias: bool = True,
        use_genotype_bias: bool = True,
        num_populations: int = 0,
        position_frequencies: int = 0,
    ) -> None:
        super().__init__()
        if position_frequencies > 0:
            self.position_encoding = FourierPositionEncoding(position_frequencies)
            position_dim = self.position_encoding.out_dim
        else:
            self.position_encoding = None
            position_dim = 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.encoder = LDAwareEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_distance_buckets=num_distance_buckets,
            max_distance=max_distance,
            genotype_rank=genotype_rank,
            use_distance_bias=use_distance_bias,
            use_genotype_bias=use_genotype_bias,
            num_populations=num_populations,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        pos_features = positions if self.position_encoding is None else self.position_encoding(positions)
        x = torch.cat([features, pos_features], dim=-1)
        x = self.input_proj(x)
        encoded, layer_attentions = self.encoder(
            x,
            positions=positions,
            attention_mask=attention_mask,
            population_id=population_id,
            need_weights=need_weights,
        )
        logits = self.classifier(encoded)
        return logits, layer_attentions
