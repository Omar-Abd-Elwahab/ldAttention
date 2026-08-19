"""Paired significance testing for the seed-level results.

The model and both controls are scored on the *same* masked entries within each
seed, so the per-seed differences are paired and deserve a paired test rather
than the overlap of two error bars.

Exact Wilcoxon signed-rank is used because the seed counts here are small (n=10)
and normality of the differences is not worth assuming. The exact null is
enumerated over all 2^n sign assignments, which is trivial at this size; above
``MAX_EXACT`` the normal approximation with a continuity correction is used
instead. scipy is deliberately not a dependency of this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

MAX_EXACT = 20


@dataclass(frozen=True)
class PairedTest:
    n: int
    n_wins: int
    mean_delta: float
    median_delta: float
    statistic: float
    p_value: float
    exact: bool

    def describe(self) -> str:
        p = "p < 0.001" if self.p_value < 0.001 else f"p = {self.p_value:.3f}"
        return f"{self.mean_delta:+.4f} mean, {self.n_wins}/{self.n} seeds, {p}"


def _rank_with_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: list[float]) -> PairedTest:
    """Two-sided exact Wilcoxon signed-rank test on paired differences."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    n_wins = sum(d > 0 for d in diffs)
    mean_delta = sum(diffs) / len(diffs) if diffs else 0.0
    ordered = sorted(diffs)
    mid = len(ordered) // 2
    median_delta = (
        ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    ) if ordered else 0.0

    if n == 0:
        return PairedTest(len(diffs), n_wins, mean_delta, median_delta, 0.0, 1.0, True)

    ranks = _rank_with_ties([abs(d) for d in nonzero])
    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks) if d < 0)
    statistic = min(w_plus, w_minus)

    if n <= MAX_EXACT:
        total = 0
        hits = 0
        for signs in product((0, 1), repeat=n):
            positive = sum(r for s, r in zip(signs, ranks) if s)
            negative = sum(ranks) - positive
            total += 1
            if min(positive, negative) <= statistic:
                hits += 1
        return PairedTest(len(diffs), n_wins, mean_delta, median_delta, statistic, hits / total, True)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (statistic - mean_w + 0.5) / math.sqrt(var_w)
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return PairedTest(len(diffs), n_wins, mean_delta, median_delta, statistic, min(p, 1.0), False)
