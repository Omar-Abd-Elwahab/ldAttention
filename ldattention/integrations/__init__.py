"""Adapters for dropping LD-aware bias into existing transformer pipelines."""

from ldattention.integrations.torch_mha import (
    LDAwareMultiheadAttention,
    ld_bias_to_mha_mask,
)

__all__ = [
    "LDAwareMultiheadAttention",
    "ld_bias_to_mha_mask",
]
