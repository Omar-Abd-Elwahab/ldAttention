"""Figures for the long-window and multi-chromosome experiments.

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

BLUE = "#2F5D9F"
GOLD = "#C8961A"
GREY = "#7A7A7A"
RED = "#B3402F"
DARK = "#222222"

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


def _ms(values: list[float]) -> tuple[float, float]:
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)


def _paired_gap(rows: list[dict]) -> tuple[float, float]:
    """Mean and sd of (ldAttention - plain transformer) over shared seeds."""
    both = {int(r["seed"]): float(r["accuracy"]) for r in rows if r["arm"] == "both"}
    plain = {int(r["seed"]): float(r["accuracy"]) for r in rows if r["arm"] == "no_bias"}
    paired = [both[s] - plain[s] for s in sorted(set(both) & set(plain))]
    return _ms(paired)


def window_figure() -> Path:
    rows = _rows("window_short/window_raw.csv", "window_long/window_raw.csv")
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_len[int(r["n_sites"])].append(r)
    lengths = sorted(by_len)
    # The per-window budget falls as L grows, so the sweep confounds window
    # length with training time. This run retrains the longest window for the
    # budget a short window got, which is what separates the two.
    converged = _rows("window_converge/window_raw.csv")

    series = {
        "ldAttention": ("accuracy", "both", BLUE, "o"),
        "Plain transformer": ("accuracy", "no_bias", GREY, "s"),
        "Explicit LD (top-64)": ("explicit_ld_top64", "both", GOLD, "^"),
        "Explicit LD (top-8)": ("explicit_ld_top8", "both", "#E0BE72", "v"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    ax = axes[0]
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
        for arm, color in (("both", BLUE), ("no_bias", GREY)):
            vals = [100 * float(r["accuracy"]) for r in converged if r["arm"] == arm]
            ax.scatter([L_c], [mean(vals)], s=150, marker="*", color=color,
                       edgecolor="white", zorder=4)
        ax.annotate("retrained to\n300 epochs", (L_c, mean(
            [100 * float(r["accuracy"]) for r in converged if r["arm"] == "no_bias"])),
            textcoords="offset points", xytext=(-14, -34), ha="right", fontsize=8.5,
            color="#555555", arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
    ax.set_xlabel("SNPs in the window")
    ax.set_ylabel("Held-out imputation accuracy (%)")
    ax.set_title("A.  Both transformers beat explicit LD at every length")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="lower left")

    ax = axes[1]
    labels, deltas, errs, colors = [], [], [], []
    for L in lengths:
        m, s = _paired_gap(by_len[L])
        labels.append(f"{L}\n{by_len[L][0]['epochs']} ep")
        deltas.append(100 * m)
        errs.append(100 * s)
        colors.append(BLUE)
    if converged:
        m, s = _paired_gap(converged)
        labels.append(f"{converged[0]['n_sites']}\n{converged[0]['epochs']} ep")
        deltas.append(100 * m)
        errs.append(100 * s)
        colors.append(RED)
    bars = ax.bar(labels, deltas, yerr=errs, capsize=4, color=colors,
                  edgecolor="white", width=0.62)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, d + max(deltas) * 0.03, f"{d:+.2f}",
                ha="center", fontsize=10, fontweight="bold", color=bar.get_facecolor())
    ax.axhline(0, color=DARK, lw=1)
    ax.set_xlabel("SNPs in the window, and epochs of training")
    ax.set_ylabel("Accuracy gain over a plain transformer (pp)")
    ax.set_title("B.  Most of that gain was undertraining, not window length")
    ax.set_ylim(0, max(deltas) * 1.32)
    ax.text(0.03, 0.96,
            "The budget shrinks as windows grow, so length and\n"
            "training time are confounded. Give 1024 SNPs the\n"
            "same budget and the gap falls from +7.08 to +0.85 pp.",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#555555", style="italic")

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
        return _ms([mean(v) for v in by_seed.values()])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    ax = axes[0]
    groups = ["Same chromosomes\n(held-out people)", "Unseen chromosome\n(zero-shot)"]
    entries = [
        ("ldAttention", "both_multi", "accuracy", BLUE),
        ("Plain transformer", "no_bias_multi", "accuracy", GREY),
        ("Explicit LD (refit)", "both_multi", "explicit_ld", GOLD),
        ("Majority genotype", "both_multi", "majority", "#BBBBBB"),
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
    ax.set_title("A.  Trained weights do not transfer across chromosomes")
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", ncol=2, loc="upper center")
    cross_model = 100 * agg("both_multi", "cross", "accuracy")[0]
    ax.annotate("below the\nmajority floor", xy=(1 - 1.5 * width, cross_model),
                xytext=(1 - 2.6 * width, cross_model + 24), fontsize=9, color=RED,
                ha="center", arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))

    ax = axes[1]
    for i, (label, field, color) in enumerate([
        ("attention vs true $r^2$", "attention_vs_r2_pearson", BLUE),
        ("after removing distance", "attention_vs_r2_partial_pearson", "#8FB0D8"),
    ]):
        vals, errs = zip(*[agg("both_multi", kind, field) for kind in ("within", "cross")])
        ax.bar(x + (i - 0.5) * 0.3, vals, 0.3, yerr=errs, capsize=3,
               color=color, edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Same chromosomes", "Unseen chromosome"])
    ax.set_ylabel("Pearson correlation")
    ax.set_title("B.  Only the generic distance prior survives")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd")
    ax.set_ylim(0, 0.62)

    fig.tight_layout()
    path = OUT / "chromosome_transfer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def cost_figure() -> Path:
    import json

    cost = json.loads((SCALE / "compute_cost.json").read_text())
    rows = {r["n_sites"]: r for r in cost["rows"]}
    lengths = sorted(rows)

    acc_rows = _rows("window_short/window_raw.csv", "window_long/window_raw.csv")
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in acc_rows:
        by_len[int(r["n_sites"])].append(r)

    def acc(L: int, field: str, arm: str) -> float:
        return 100 * mean([float(r[field]) for r in by_len[L] if r["arm"] == arm])

    methods = [
        ("ldAttention", "ldattention_train_total_s", "accuracy", "both", BLUE, "o"),
        ("Plain transformer", "plain_train_total_s", "accuracy", "no_bias", GREY, "s"),
        ("Explicit LD (top-64)", "explicit_top64_fit_total_s", "explicit_ld_top64", "both", GOLD, "^"),
        ("Explicit LD (top-8)", "explicit_top8_fit_total_s", "explicit_ld_top8", "both", "#E0BE72", "v"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    ax = axes[0]
    for label, tfield, _afield, _arm, color, marker in methods:
        mins = [rows[L][tfield] / 60 for L in lengths]
        ax.plot(lengths, mins, marker=marker, color=color, lw=2.2, ms=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("SNPs in the window")
    ax.set_ylabel("Time to fit one cohort (min, log scale)")
    ax.set_title("A.  Fitting cost")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.5)

    ax = axes[1]
    L = max(lengths)
    offsets = {"ldAttention": (0, 14), "Plain transformer": (0, -22),
               "Explicit LD (top-64)": (14, 10), "Explicit LD (top-8)": (14, -18)}
    for label, tfield, afield, arm, color, marker in methods:
        x, y = rows[L][tfield] / 60, acc(L, afield, arm)
        ax.scatter(x, y, s=130, color=color, marker=marker, edgecolor="white", zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=offsets[label], ha="left" if offsets[label][0] else "center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 400)
    ax.set_xlabel("Time to fit one cohort (min, log scale)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title(f"B.  What the compute buys at {L} SNPs")
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.set_ylim(86, 102)

    ax = axes[2]
    transformer = [rows[L]["ldattention_artifact_bytes"] / 2**20 for L in lengths]
    # The r^2 table is transient preprocessing state, but it is the L x L object
    # the layer exists to avoid, so it is shown separately from the fitted model.
    r2_tab = [rows[L]["r2_bytes"] / 2**20 for L in lengths]
    reg = [(rows[L]["explicit_top8_artifact_bytes"] - rows[L]["r2_bytes"]) / 2**20 for L in lengths]
    ax.plot(lengths, transformer, marker="o", color=BLUE, lw=2.2, ms=7, label="ldAttention weights")
    ax.plot(lengths, reg, marker="v", color="#E0BE72", lw=2.2, ms=7, label="Explicit LD weights (top-8)")
    ax.plot(lengths, r2_tab, marker="^", color=GOLD, lw=2.2, ms=7, label="$r^2$ table ($L^2$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("SNPs in the window")
    ax.set_ylabel("Stored / built artifact (MB, log scale)")
    ax.set_title("C.  What has to be stored")
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    fig.tight_layout()
    path = OUT / "cost_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (window_figure(), chromosome_figure(), cost_figure()):
        print(p)
