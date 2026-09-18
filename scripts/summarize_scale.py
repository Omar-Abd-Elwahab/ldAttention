"""Summarize the long-window and multi-chromosome sweeps.

Reads the per-run CSVs written by ``run_scale_experiments.py`` (any number of
output directories, so a sweep split across several invocations can be pooled)
and reports paired, per-seed comparisons with exact Wilcoxon tests.

    .venv312/bin/python3.12 scripts/summarize_scale.py results_scale/*
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.stats import wilcoxon_signed_rank  # noqa: E402


def _read(paths: list[Path], filename: str) -> list[dict]:
    rows: list[dict] = []
    for directory in paths:
        path = directory / filename
        if path.exists():
            rows.extend(csv.DictReader(path.open()))
    return rows


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _pm(values: list[float], scale: float = 100.0, digits: int = 2) -> str:
    if not values:
        return "n/a"
    m = mean(values) * scale
    s = (pstdev(values) if len(values) > 1 else 0.0) * scale
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _paired(a: dict[int, float], b: dict[int, float]) -> tuple[list[float], object]:
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    return diffs, wilcoxon_signed_rank(diffs) if diffs else None


def summarize_window(rows: list[dict]) -> None:
    if not rows:
        return
    print("=" * 108)
    print("WINDOW LENGTH  (accuracy %, mean ± sd across seeds; paired exact Wilcoxon on per-seed deltas)")
    print("=" * 108)
    by_window: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_window[int(r["n_sites"])].append(r)

    header = (f"{'L':>5} {'span kb':>8} {'ldAttention':>15} {'plain tfmr':>15} "
              f"{'LD top-8':>15} {'LD top-64':>15} {'majority':>15}")
    print(header)
    print("-" * len(header))
    for L in sorted(by_window):
        rs = by_window[L]
        both = [r for r in rs if r["arm"] == "both"]
        plain = [r for r in rs if r["arm"] == "no_bias"]
        span = mean([_f(r, "span_bp") for r in rs]) / 1000.0
        print(f"{L:5d} {span:8.0f} "
              f"{_pm([_f(r, 'accuracy') for r in both]):>15} "
              f"{_pm([_f(r, 'accuracy') for r in plain]):>15} "
              f"{_pm([_f(r, 'explicit_ld_top8') for r in both]):>15} "
              f"{_pm([_f(r, 'explicit_ld_top64') for r in both]):>15} "
              f"{_pm([_f(r, 'majority') for r in both]):>15}")

    print("\nPaired deltas (percentage points), ldAttention minus control")
    sub = f"{'L':>5} {'vs top-8':>22} {'vs top-64':>22} {'vs plain transformer':>26}"
    print(sub)
    print("-" * len(sub))
    for L in sorted(by_window):
        rs = by_window[L]
        both = {int(r["seed"]): _f(r, "accuracy") for r in rs if r["arm"] == "both"}
        plain = {int(r["seed"]): _f(r, "accuracy") for r in rs if r["arm"] == "no_bias"}
        top8 = {int(r["seed"]): _f(r, "explicit_ld_top8") for r in rs if r["arm"] == "both"}
        top64 = {int(r["seed"]): _f(r, "explicit_ld_top64") for r in rs if r["arm"] == "both"}
        cells = []
        for control in (top8, top64, plain):
            diffs, test = _paired(both, control)
            if test is None:
                cells.append("n/a")
                continue
            cells.append(f"{100 * test.mean_delta:+.2f} ({test.n_wins}/{test.n}, p={test.p_value:.3f})")
        print(f"{L:5d} {cells[0]:>22} {cells[1]:>22} {cells[2]:>26}")

    print("\nLD recovery and cost")
    sub = (f"{'L':>5} {'attn~r2 (both)':>16} {'partial':>16} {'attn~r2 (plain)':>16} "
           f"{'peak GB':>9} {'train min':>10} {'epochs':>7}")
    print(sub)
    print("-" * len(sub))
    for L in sorted(by_window):
        rs = by_window[L]
        both = [r for r in rs if r["arm"] == "both"]
        plain = [r for r in rs if r["arm"] == "no_bias"]
        print(f"{L:5d} "
              f"{_pm([_f(r, 'attention_vs_r2_pearson') for r in both], 1.0, 3):>16} "
              f"{_pm([_f(r, 'attention_vs_r2_partial_pearson') for r in both], 1.0, 3):>16} "
              f"{_pm([_f(r, 'attention_vs_r2_pearson') for r in plain], 1.0, 3):>16} "
              f"{max(_f(r, 'peak_gb') for r in both):9.2f} "
              f"{mean([_f(r, 'train_seconds') for r in both]) / 60:10.1f} "
              f"{int(both[0]['epochs']):7d}")


def summarize_chromosome(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 108)
    print("CHROMOSOME TRANSFER  (accuracy %, held-out individuals; 'cross' = chromosome never seen in training)")
    print("=" * 108)

    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "both_multi", a))
    header = f"{'arm':>15} {'kind':>7} {'ldAttention/plain':>19} {'explicit LD (refit)':>21} {'majority':>15} {'delta pp':>22}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        for kind in ("within", "cross"):
            rs = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            if not rs:
                continue
            # Average over chromosomes within a seed first: chromosomes of one
            # seed share a cohort and a trained model, so they are not
            # independent replicates.
            by_seed: dict[int, list[dict]] = defaultdict(list)
            for r in rs:
                by_seed[int(r["seed"])].append(r)
            model = {s: mean([_f(r, "accuracy") for r in v]) for s, v in by_seed.items()}
            explicit = {s: mean([_f(r, "explicit_ld") for r in v]) for s, v in by_seed.items()}
            majority = [mean([_f(r, "majority") for r in v]) for v in by_seed.values()]
            diffs, test = _paired(model, explicit)
            delta = (f"{100 * test.mean_delta:+.2f} ({test.n_wins}/{test.n}, p={test.p_value:.3f})"
                     if test else "n/a")
            print(f"{arm:>15} {kind:>7} {_pm(list(model.values())):>19} "
                  f"{_pm(list(explicit.values())):>21} {_pm(majority):>15} {delta:>22}")

    print("\nAttention vs the target chromosome's own r^2 (never an input)")
    sub = f"{'arm':>15} {'kind':>7} {'pearson r':>16} {'distance-controlled':>21}"
    print(sub)
    print("-" * len(sub))
    for arm in arms:
        for kind in ("within", "cross"):
            rs = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            if not rs:
                continue
            print(f"{arm:>15} {kind:>7} "
                  f"{_pm([_f(r, 'attention_vs_r2_pearson') for r in rs], 1.0, 3):>16} "
                  f"{_pm([_f(r, 'attention_vs_r2_partial_pearson') for r in rs], 1.0, 3):>21}")

    print("\nCross-chromosome cost of the LD bias (ldAttention minus plain transformer, pooled training)")
    for kind in ("within", "cross"):
        both = defaultdict(list)
        plain = defaultdict(list)
        for r in rows:
            if r["kind"] != kind:
                continue
            if r["arm"] == "both_multi":
                both[int(r["seed"])].append(_f(r, "accuracy"))
            elif r["arm"] == "no_bias_multi":
                plain[int(r["seed"])].append(_f(r, "accuracy"))
        a = {s: mean(v) for s, v in both.items()}
        b = {s: mean(v) for s, v in plain.items()}
        diffs, test = _paired(a, b)
        if test:
            print(f"  {kind:>6}: {100 * test.mean_delta:+.2f} pp "
                  f"({test.n_wins}/{test.n} seeds, p={test.p_value:.3f})")

    spans = sorted({round(_f(r, "span_bp") / 1000) for r in rows})
    print(f"\nPhysical spans of the simulated chromosomes (kb): {spans}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    args = p.parse_args()
    dirs = [d for d in args.dirs if d.is_dir()]
    summarize_window(_read(dirs, "window_raw.csv"))
    summarize_chromosome(_read(dirs, "chrom_raw.csv"))


if __name__ == "__main__":
    main()
