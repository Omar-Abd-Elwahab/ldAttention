"""Drop-in adapters for stock PyTorch attention.

The whole point of :class:`~ldattention.models.ld_bias.LDAttentionBias` is that
it emits a plain additive bias tensor, so it can be injected into attention that
you did not write. This module shows two integration points:

- :func:`ld_bias_to_mha_mask` reshapes the ``[B, H, L, L]`` bias into the
  ``[B*H, L, L]`` additive ``attn_mask`` expected by
  :class:`torch.nn.MultiheadAttention`.
- :class:`LDAwareMultiheadAttention` wraps a stock
  :class:`torch.nn.MultiheadAttention` and feeds it the LD bias, giving you an
  LD-aware layer without reimplementing attention.

For ``torch.nn.functional.scaled_dot_product_attention`` no adapter is needed:
just pass the bias directly as ``attn_mask=bias``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ldattention.models.ld_bias import LDAttentionBias


def ld_bias_to_mha_mask(bias: torch.Tensor) -> torch.Tensor:
    """Reshape ``[B, H, L, L]`` LD bias to the ``[B*H, L, S]`` MHA ``attn_mask``.

    ``torch.nn.MultiheadAttention`` accepts a 3D float ``attn_mask`` of shape
    ``(N * num_heads, L, S)`` that is added to the attention logits.
    """
    if bias.dim() != 4:
        raise ValueError("bias must be [B, H, L, L]")
    batch_size, num_heads, q_len, k_len = bias.shape
    return bias.reshape(batch_size * num_heads, q_len, k_len)


class LDAwareMultiheadAttention(nn.Module):
    """LD-aware wrapper around :class:`torch.nn.MultiheadAttention`.

    This performs standard multi-head self-attention but adds the learned
    LD-aware bias to the attention logits, so any pipeline already using
    ``nn.MultiheadAttention`` can become LD-aware by swapping this in.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        **ld_bias_kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.ld_bias = LDAttentionBias(embed_dim, num_heads, **ld_bias_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        population_id: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: ``[B, L, E]`` if ``batch_first`` else ``[L, B, E]``.
            positions: ``[B, L]`` or ``[B, L, 1]`` genomic positions.
            key_padding_mask: ``[B, L]`` where 1/True means keep (this is the
                inverse of ``nn.MultiheadAttention``'s convention, and is
                converted internally).
            population_id: optional ``[B]`` population indices.
            need_weights: return per-head attention weights.
        """
        if not self.batch_first:
            # LD bias is computed in batch-first layout.
            x_bf = x.transpose(0, 1)
        else:
            x_bf = x

        # Fold padding into the additive bias (rather than passing a separate
        # bool key_padding_mask) so both masks share the float type MHA expects.
        bias = self.ld_bias(
            positions=positions,
            token_embeddings=x_bf,
            key_padding_mask=key_padding_mask,
            population_id=population_id,
        )
        attn_mask = ld_bias_to_mha_mask(bias).to(x.dtype)

        out, weights = self.mha(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return out, weights
