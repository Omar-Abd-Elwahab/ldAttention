"""Reference imputation baselines for LD-aware attention.

Two controls are provided:

``majority_baseline``
    Per-site modal genotype from the training split. Establishes the accuracy
    floor set by allele frequency alone.

``LDRegressionBaseline``
    The *explicit-LD pipeline* that ``ldAttention`` aims to replace. It first
    materializes the full pairwise ``r^2`` matrix on the training split, keeps
    the top-``k`` LD partners per target site, and fits a per-site multinomial
    logistic regression over those partners' observed alleles. The ``r^2`` build
    cost is timed separately so the preprocessing burden can be reported.

    It is given *exactly* the same per-individual input tensor as the model
    (both phased alleles plus the missingness indicator); the only difference is
    that it reaches its partners through an explicit ``r^2`` ranking instead of
    through learned attention. The head-to-head is therefore about how LD
    structure is discovered, not about how much data each method sees.

Both controls are scored on exactly the same masked entries as the model, so
comparisons are paired.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def majority_baseline(
    train_labels: torch.Tensor,
    eval_labels: torch.Tensor,
    masks: list[torch.Tensor],
    num_classes: int = 3,
) -> dict[str, float | np.ndarray]:
    """Predict each site's modal training genotype -- the allele-frequency floor."""
    counts = torch.stack([(train_labels == c).sum(dim=0) for c in range(num_classes)])
    modal = counts.argmax(dim=0)  # [L]
    flags: list[torch.Tensor] = []
    sites: list[torch.Tensor] = []
    for mask in masks:
        pred = modal.unsqueeze(0).expand_as(eval_labels)
        flags.append((pred[mask] == eval_labels[mask]).to(torch.float32))
        sites.append(mask.nonzero(as_tuple=True)[1])
    correct = torch.cat(flags).cpu().numpy()
    return {
        "accuracy": float(correct.mean()) if correct.size else 0.0,
        "correct": correct,
        "sites": torch.cat(sites).cpu().numpy(),
    }


def select_ld_partners(r2: np.ndarray, top_k: int) -> np.ndarray:
    """Indices of the ``top_k`` highest-``r^2`` partners for every site."""
    scores = r2.copy()
    np.fill_diagonal(scores, -np.inf)
    k = min(top_k, scores.shape[0] - 1)
    return np.argsort(-scores, axis=1)[:, :k]


class LDRegressionBaseline:
    """Per-site multinomial logistic regression over explicit top-``r^2`` partners.

    Input per target site is the observed state of its ``k`` explicit LD partners:
    both alleles (as the model sees them) plus a missingness indicator, so masked
    partners are distinguishable from genuine homozygous-reference calls.
    """

    def __init__(
        self,
        partner_idx: np.ndarray,
        device: torch.device,
        num_classes: int = 3,
        lr: float = 5e-2,
        epochs: int = 120,
        weight_decay: float = 1e-4,
    ) -> None:
        self.partner_idx = torch.as_tensor(partner_idx, dtype=torch.long, device=device)
        n_sites, k = self.partner_idx.shape
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        n_features = 3 * k  # allele 1, allele 2, missing-flag per partner
        self.weight = torch.zeros(n_sites, n_features, num_classes, device=device, requires_grad=True)
        self.bias = torch.zeros(n_sites, num_classes, device=device, requires_grad=True)

    def _logits(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``features`` [B, L, 2] phased alleles, ``mask`` [B, L] bool -> logits [B, L, C]."""
        observed = features.masked_fill(mask.unsqueeze(-1), 0.0)
        missing = mask.to(features.dtype).unsqueeze(-1)
        state = torch.cat([observed, missing], dim=-1)  # [B, L, 3]
        # Gather each site's partners: [B, L, k, 3] -> flattened [B, L, 3k].
        partners = state[:, self.partner_idx]
        partners = partners.reshape(*partners.shape[:2], -1)
        return torch.einsum("blf,lfc->blc", partners, self.weight) + self.bias

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        mask_rate: float,
        generator: torch.Generator,
        batch_size: int = 128,
        block_missing: bool = False,
        block_len: int = 8,
    ) -> None:
        from ldattention.validation import sample_mask

        optimizer = torch.optim.AdamW([self.weight, self.bias], lr=self.lr, weight_decay=self.weight_decay)
        n = train_labels.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=generator, device=train_labels.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                feats, labs = train_features[idx], train_labels[idx]
                mask = sample_mask(
                    labs.shape[0], labs.shape[1], mask_rate, generator, labs.device,
                    block_missing, block_len,
                )
                if not bool(mask.any()):
                    continue
                logits = self._logits(feats, mask)
                loss = F.cross_entropy(logits[mask], labs[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def score(
        self,
        eval_features: torch.Tensor,
        eval_labels: torch.Tensor,
        masks: list[torch.Tensor],
    ) -> dict[str, float | np.ndarray]:
        """Paired scoring on the same masked entries the model is scored on."""
        flags: list[torch.Tensor] = []
        sites: list[torch.Tensor] = []
        for mask in masks:
            pred = self._logits(eval_features, mask).argmax(dim=-1)
            flags.append((pred[mask] == eval_labels[mask]).to(torch.float32))
            sites.append(mask.nonzero(as_tuple=True)[1])
        correct = torch.cat(flags).cpu().numpy()
        return {
            "accuracy": float(correct.mean()) if correct.size else 0.0,
            "correct": correct,
            "sites": torch.cat(sites).cpu().numpy(),
        }


def time_explicit_r2(haplotypes: np.ndarray, repeats: int = 3) -> float:
    """Median wall-clock seconds to materialize the pairwise ``r^2`` matrix."""
    from ldattention.validation import compute_true_r2

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute_true_r2(haplotypes)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))
