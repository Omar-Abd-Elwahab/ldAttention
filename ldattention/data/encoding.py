"""Shared genotype encoding helpers."""

from __future__ import annotations

import numpy as np
import torch


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return positions
    min_pos = positions.min()
    max_pos = positions.max()
    if max_pos == min_pos:
        return np.zeros_like(positions, dtype=np.float32)
    return ((positions - min_pos) / (max_pos - min_pos)).astype(np.float32)


def genotype_to_class(a1: int, a2: int) -> int:
    if a1 == 0 and a2 == 0:
        return 0
    if (a1 == 0 and a2 == 1) or (a1 == 1 and a2 == 0):
        return 1
    if a1 == 1 and a2 == 1:
        return 2
    return -1


def make_tensors(
    allele_features: np.ndarray,
    normalized_positions: np.ndarray,
    labels: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(allele_features, dtype=torch.float32)
    positions = torch.tensor(normalized_positions[..., None], dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    return features, positions, targets
