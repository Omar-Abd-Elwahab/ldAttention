"""LD-aware transformer encoder stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from ldattention.models.ld_attention import LDAwareSelfAttention


class LDAwareEncoderLayer(nn.Module):
    """Single encoder block with LD-aware attention and MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_distance_buckets: int = 64,
        max_distance: float = 1.0,
        genotype_rank: int = 16,
        use_distance_bias: bool = True,
        use_genotype_bias: bool = True,
        num_populations: int = 0,
    ) -> None:
        super().__init__()
        self.attn = LDAwareSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_distance_buckets=num_distance_buckets,
            max_distance=max_distance,
            genotype_rank=genotype_rank,
            use_distance_bias=use_distance_bias,
            use_genotype_bias=use_genotype_bias,
            num_populations=num_populations,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(
            self.norm1(x),
            positions,
            attention_mask=attention_mask,
            population_id=population_id,
            need_weights=need_weights,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights


class LDAwareEncoder(nn.Module):
    """Stacked LD-aware encoder layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        num_distance_buckets: int = 64,
        max_distance: float = 1.0,
        genotype_rank: int = 16,
        use_distance_bias: bool = True,
        use_genotype_bias: bool = True,
        num_populations: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LDAwareEncoderLayer(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    num_distance_buckets=num_distance_buckets,
                    max_distance=max_distance,
                    genotype_rank=genotype_rank,
                    use_distance_bias=use_distance_bias,
                    use_genotype_bias=use_genotype_bias,
                    num_populations=num_populations,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        layer_attentions: list[torch.Tensor | None] = []
        for layer in self.layers:
            x, attn_weights = layer(
                x,
                positions,
                attention_mask=attention_mask,
                population_id=population_id,
                need_weights=need_weights,
            )
            layer_attentions.append(attn_weights)
        return self.final_norm(x), layer_attentions
