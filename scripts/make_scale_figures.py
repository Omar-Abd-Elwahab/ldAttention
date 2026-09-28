"""Figures for the long-window and multi-chromosome experiments.

Prefer equal-budget window results when present; otherwise fall back to the
legacy short/long/converge CSVs. Labels, axis titles and captions stay literal
(accuracy %, minutes, MB) — no marketing jargon.

    .venv312/bin/python3.12 scripts/make_scale_figures.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = ROOT / "results_scale"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette across all manuscript figures (keep in sync with make_figures.py).
BLUE = "#2F5D9F"        # LDAttention + Bias (primary)
SLATE = "#5B7C99"       # LDAttention alone
GOLD = "#C8961A"        # Explicit LD (top-64 / all partners / main)
GOLD_LIGHT = "#E0BE72"  # Explicit LD (top-8)
MAJORITY = "#A8A8A8"    # Majority genotype baseline
RED = "#B3402F"
DARK = "#222222"
GREY = SLATE  # back-compat alias
BLUE_LIGHT = "#8FB0D8"  # secondary attention-correlation bars

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#666666", "figure.facecolor": "white",
})


def _rows(*names: str) -> list[dict]:
    out: list[dict] = []
    for name in names:
        path = SCALE / name
        if path.exists():
            out.extend(csv.DictReader(path.open()))
    return out


def _window_rows() -> tuple[list[dict], str]:
    """Prefer the equal-budget sweep; otherwise the legacy multi-budget CSVs."""
    equal = _rows("window_equal/window_raw.csv")
    if equal:
        return equal, "equal"
    legacy = _rows("window_short/window_raw.csv", "window_long/window_raw.csv")
    return legacy, "legacy"


def _ms(values: list[float]) -> tuple[float, float]:
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)


def _paired_gap(rows: list[dict]) -> tuple[float, float]:
    """Mean and sd of (ldAttention - plain transformer) over shared seeds."""
    both = {int(r["seed"]): float(r["accuracy"]) for r in rows if r["arm"] == "both"}
    plain = {int(r["seed"]): float(r["accuracy"]) for r in rows if r["arm"] == "no_bias"}
    paired = [both[s] - plain[s] for s in sorted(set(both) & set(plain))]
    return _ms(paired) if paired else (0.0, 0.0)


def window_figure() -> Path:
    rows, mode = _window_rows()
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_len[int(r["n_sites"])].append(r)
    lengths = sorted(by_len)
    converged = _rows("window_converge/window_raw.csv") if mode == "legacy" else []

    series = {
        "LDAttention+Bias": ("accuracy", "both", BLUE, "o"),
        "LDAttention": ("accuracy", "no_bias", SLATE, "s"),
        "Explicit LD (top-64)": ("explicit_ld_top64", "both", GOLD, "^"),
        "Explicit LD (top-8)": ("explicit_ld_top8", "both", GOLD_LIGHT, "v"),
        "Majority genotype": ("majority", "both", MAJORITY, "D"),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, (field, arm, color, marker) in series.items():
        m, s = zip(*[_ms([float(r[field]) for r in by_len[L] if r["arm"] == arm]) for L in lengths])
        m, s = 100 * np.array(m), 100 * np.array(s)
        ax.plot(lengths, m, marker=marker, color=color, lw=2.2, ms=7, label=label)
        ax.fill_between(lengths, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    if converged:
        L_c = int(converged[0]["n_sites"])
        for arm, color in (("both", BLUE), ("no_bias", SLATE)):
            vals = [100 * float(r["accuracy"]) for r in converged if r["arm"] == arm]
            if vals:
                ax.scatter([L_c], [mean(vals)], s=150, marker="*", color=color,
                           edgecolor="white", zorder=4)
    ax.set_xlabel("Number of SNPs in the window ($L$)")
    ax.set_ylabel("Held-out genotype accuracy (%)")
    if mode == "equal":
        epochs = int(by_len[lengths[0]][0]["epochs"])
        ax.set_title(f"Accuracy vs window length ({epochs} epochs, matched budget)")
    else:
        ax.set_title("Accuracy vs window length")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="center right")
    ax.set_ylim(bottom=min(60, ax.get_ylim()[0]))

    fig.tight_layout()
    path = OUT / "window_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def chromosome_figure() -> Path:
    rows = [r for r in _rows("chrom/chrom_raw.csv") if r["arm"] in ("both_multi", "no_bias_multi")]

    def agg(arm: str, kind: str, field: str) -> tuple[float, float]:
        by_seed = defaultdict(list)
        for r in rows:
            if r["arm"] == arm and r["kind"] == kind:
                by_seed[int(r["seed"])].append(float(r[field]))
        return _ms([mean(v) for v in by_seed.values()]) if by_seed else (0.0, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    ax = axes[0]
    groups = ["Same chromosomes\n(held-out people)", "Unseen chromosome\n(zero-shot)"]
    entries = [
        ("LDAttention+Bias", "both_multi", "accuracy", BLUE),
        ("LDAttention", "no_bias_multi", "accuracy", SLATE),
        ("Explicit LD (refit)", "both_multi", "explicit_ld", GOLD),
        ("Majority genotype", "both_multi", "majority", MAJORITY),
    ]
    width, x = 0.2, np.arange(2)
    for i, (label, arm, field, color) in enumerate(entries):
        vals, errs = zip(*[agg(arm, kind, field) for kind in ("within", "cross")])
        ax.bar(x + (i - 1.5) * width, 100 * np.array(vals), width,
               yerr=100 * np.array(errs), capsize=3, color=color,
               edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Held-out imputation accuracy (%)")
    ax.set_title("A.  Within-chromosome fit vs zero-shot transfer")
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", ncol=2, loc="upper center")
    cross_model = 100 * agg("both_multi", "cross", "accuracy")[0]
    ax.annotate("below majority\nbaseline", xy=(1 - 1.5 * width, cross_model),
                xytext=(1 - 2.6 * width, cross_model + 24), fontsize=9, color=RED,
                ha="center", arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))

    ax = axes[1]
    for i, (label, field, color) in enumerate([
        ("Attention vs true $r^2$", "attention_vs_r2_pearson", BLUE),
        ("Partial (distance removed)", "attention_vs_r2_partial_pearson", BLUE_LIGHT),
    ]):
        vals, errs = zip(*[agg("both_multi", kind, field) for kind in ("within", "cross")])
        ax.bar(x + (i - 0.5) * 0.3, vals, 0.3, yerr=errs, capsize=3,
               color=color, edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Same chromosomes", "Unseen chromosome"])
    ax.set_ylabel("Pearson correlation")
    ax.set_title("B.  Attention–LD alignment within and across chromosomes")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd")
    ax.set_ylim(0, 0.62)

    fig.tight_layout()
    path = OUT / "chromosome_transfer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def cost_figure() -> Path:
    import json

    cost_path = SCALE / "compute_cost.json"
    if not cost_path.exists():
        return OUT / "cost_comparison.png"
    cost = json.loads(cost_path.read_text())
    rows = {r["n_sites"]: r for r in cost["rows"]}
    lengths = sorted(rows)

    acc_rows, _mode = _window_rows()
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in acc_rows:
        by_len[int(r["n_sites"])].append(r)

    def acc(L: int, field: str, arm: str) -> float:
        vals = [float(r[field]) for r in by_len.get(L, []) if r["arm"] == arm]
        return 100 * mean(vals) if vals else float("nan")

    methods = [
        ("LDAttention+Bias", "ldattention_train_total_s", "accuracy", "both", BLUE, "o"),
        ("LDAttention", "plain_train_total_s", "accuracy", "no_bias", SLATE, "s"),
        ("Explicit LD (top-64)", "explicit_top64_fit_total_s", "explicit_ld_top64", "both", GOLD, "^"),
        ("Explicit LD (top-8)", "explicit_top8_fit_total_s", "explicit_ld_top8", "both", GOLD_LIGHT, "v"),
        ("Majority genotype", None, "majority", "both", MAJORITY, "D"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    ax = axes[0]
    for label, tfield, _afield, _arm, color, marker in methods:
        if tfield is None:
            # Majority fit is effectively instantaneous relative to model training.
            mins = [0.01 for _ in lengths]
        else:
            mins = [rows[L][tfield] / 60 for L in lengths]
        ax.plot(lengths, mins, marker=marker, color=color, lw=2.2, ms=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Number of SNPs ($L$)")
    ax.set_ylabel("Wall-clock fit time (min, log scale)")
    ax.set_title("A.  Fit time vs window length")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    ax = axes[1]
    L = max(lengths)
    offsets = {
        "LDAttention+Bias": (-8, 12),
        "LDAttention": (8, -18),
        "Explicit LD (top-64)": (10, 12),
        "Explicit LD (top-8)": (10, -16),
        "Majority genotype": (-10, -14),
    }
    ys = []
    xs = []
    for label, tfield, afield, arm, color, marker in methods:
        x = 0.01 if tfield is None else rows[L][tfield] / 60
        y = acc(L, afield, arm)
        if np.isnan(y):
            continue
        ys.append(y)
        xs.append(x)
        ax.scatter(x, y, s=130, color=color, marker=marker, edgecolor="white", zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=offsets[label], ha="right" if offsets[label][0] < 0 else "left",
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.005, max(xs) * 4 if xs else 1)
    if ys:
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_xlabel("Wall-clock fit time (min, log scale)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title(f"B.  Accuracy–cost trade-off at $L$={L}")
    ax.grid(alpha=0.25, which="both", lw=0.5)

    ax = axes[2]
    # Storage order: transformer weights are independent of L; the r^2 table is O(L^2).
    transformer = [rows[L]["ldattention_artifact_bytes"] / 2**20 for L in lengths]
    r2_tab = [rows[L]["r2_bytes"] / 2**20 for L in lengths]
    reg = [(rows[L]["explicit_top8_artifact_bytes"] - rows[L]["r2_bytes"]) / 2**20 for L in lengths]
    ax.plot(lengths, transformer, marker="o", color=BLUE, lw=2.2, ms=7,
            label="LDAttention weights ($O(1)$ in $L$)")
    ax.plot(lengths, reg, marker="v", color=GOLD_LIGHT, lw=2.2, ms=7,
            label="Explicit LD weights (top-8)")
    ax.plot(lengths, r2_tab, marker="^", color=GOLD, lw=2.2, ms=7,
            label="$r^2$ table ($O(L^2)$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Number of SNPs ($L$)")
    ax.set_ylabel("Stored artifact size (MB, log scale)")
    ax.set_title("C.  Storage: constant weights vs quadratic $r^2$ table")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    fig.tight_layout()
    path = OUT / "cost_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def advantage_summary_figure() -> Path:
    """One panel that states the three supported advantages without false claims."""
    import json

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # Panel A: absolute accuracy of all five methods (same palette as Fig. 2).
    rows, mode = _window_rows()
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_len[int(r["n_sites"])].append(r)
    lengths = sorted(by_len)
    ax = axes[0]
    if lengths:
        series = (
            ("LDAttention+Bias", "accuracy", "both", BLUE, "o"),
            ("LDAttention", "accuracy", "no_bias", SLATE, "s"),
            ("Explicit LD (top-64)", "explicit_ld_top64", "both", GOLD, "^"),
            ("Explicit LD (top-8)", "explicit_ld_top8", "both", GOLD_LIGHT, "v"),
            ("Majority genotype", "majority", "both", MAJORITY, "D"),
        )
        for label, field, arm, color, marker in series:
            m, s = zip(*[_ms([float(r[field]) for r in by_len[L] if r["arm"] == arm]) for L in lengths])
            m, s = 100 * np.array(m), 100 * np.array(s)
            ax.plot(lengths, m, marker=marker, color=color, lw=2.0, ms=6, label=label)
            ax.fill_between(lengths, m - s, m + s, color=color, alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(L) for L in lengths])
        ax.set_xlabel("Window length $L$")
        ax.set_ylabel("Held-out accuracy (%)")
        title = "A.  Accuracy vs window length" + (" (equal budget)" if mode == "equal" else "")
        ax.set_title(title)
        ax.set_ylim(60, 101)
        ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", fontsize=7.5, loc="center right")
    else:
        ax.set_title("A.  Accuracy (no data yet)")
        ax.set_axis_off()

    # Panel B: no r^2 preprocess for the neural model
    ax = axes[1]
    cost_path = SCALE / "compute_cost.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
        crow = {r["n_sites"]: r for r in cost["rows"]}
        lengths_c = sorted(crow)
        ax.bar(
            [str(L) for L in lengths_c],
            [crow[L]["r2_build_cpu_s"] for L in lengths_c],
            color=GOLD, edgecolor="white", label="Explicit pipeline: build $r^2$",
        )
        ax.bar(
            [str(L) for L in lengths_c],
            [0.0 for _ in lengths_c],
            color=BLUE, edgecolor="white", label="LDAttention: none",
        )
        ax.set_xlabel("Window length $L$")
        ax.set_ylabel("Preprocess time before fitting (s)")
        ax.set_title("B.  No pairwise $r^2$ table to build")
        ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", fontsize=8.5)
    else:
        ax.set_title("B.  No pairwise $r^2$ table (bench pending)")
        ax.set_axis_off()

    # Panel C: storage order O(1) vs O(L^2)
    ax = axes[2]
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
        crow = {r["n_sites"]: r for r in cost["rows"]}
        lengths_c = sorted(crow)
        ax.plot(lengths_c, [crow[L]["ldattention_artifact_bytes"] / 2**20 for L in lengths_c],
                marker="o", color=BLUE, lw=2.2, label="Model weights")
        ax.plot(lengths_c, [crow[L]["r2_bytes"] / 2**20 for L in lengths_c],
                marker="^", color=GOLD, lw=2.2, label="$r^2$ table")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lengths_c)
        ax.set_xticklabels([str(L) for L in lengths_c])
        ax.set_xlabel("Window length $L$")
        ax.set_ylabel("Bytes stored (MB, log)")
        ax.set_title("C.  Storage order: $O(1)$ weights vs $O(L^2)$ table")
        ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", fontsize=8.5)
        ax.grid(alpha=0.25, which="both", lw=0.5)
    else:
        ax.set_title("C.  Storage order (bench pending)")
        ax.set_axis_off()

    fig.suptitle(
        "Supported advantages of LDAttention  "
        "(dense attention remains $O(L^2)$; we do not claim linear attention)",
        fontsize=11, y=1.02, color="#333333",
    )
    fig.tight_layout()
    path = OUT / "ldattention_advantages.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (window_figure(), chromosome_figure(), cost_figure(), advantage_summary_figure()):
        print(p)
