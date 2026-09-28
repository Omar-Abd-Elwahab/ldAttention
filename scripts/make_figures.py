"""Publication figures from the reported experiment directories.

Writes PNGs to ``docs/figures/``. Defaults to ``results_large/`` when that
sweep exists, otherwise ``results/``. Override with ``LDATTENTION_RESULTS``.

    python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = ROOT / "results_large"
if not (_DEFAULT / "results.json").exists():
    _DEFAULT = ROOT / "results"
RESULTS = Path(os.environ.get("LDATTENTION_RESULTS", _DEFAULT))
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette across all manuscript figures.
BLUE = "#2F5D9F"        # LDAttention + Bias (primary)
SLATE = "#5B7C99"       # LDAttention alone
GOLD = "#C8961A"        # Explicit LD (top-64 / all partners / main)
GOLD_LIGHT = "#E0BE72"  # Explicit LD (top-8)
MAJORITY = "#A8A8A8"    # Majority genotype baseline
GREEN = "#3D7A4A"       # Architecture diagram only
DARK = "#222222"
GREY = SLATE  # back-compat alias used by older call sites

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_summary(root: Path = RESULTS) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    path = root / "results_summary.csv"
    if not path.exists():
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r["experiment"], r["config"])
            rows.setdefault(key, {})[r["metric"]] = float(r["mean"])
            rows[key][r["metric"] + "_std"] = float(r["std"])
    return rows


def eval_rate() -> float:
    base = json.loads((RESULTS / "config.json").read_text())["base"]
    return float(base.get("eval_mask_rate") or base["mask_rate"])


def load_strong_baseline() -> dict[tuple[str, str, float], dict]:
    path = RESULTS / "strong_baseline.csv"
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str, float], dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    meta: dict[tuple[str, str, float], dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["experiment"], row["config"], float(row["mask_rate"]))
            for field in ("accuracy", "accuracy_maf_low", "accuracy_maf_mid", "accuracy_maf_high"):
                value = row.get(field)
                if value not in (None, "", "nan"):
                    grouped[key][field].append(float(value))
            meta[key] = {"top_k": int(row["top_k"]), "epochs": int(row["epochs"])}
    out: dict[tuple[str, str, float], dict] = {}
    for key, fields in grouped.items():
        entry = dict(meta[key])
        for field, values in fields.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            entry[field] = mean
            entry[f"{field}_std"] = var ** 0.5
        out[key] = entry
    return out


def load_scaling() -> list[tuple[int, dict[str, tuple[float, float]]]]:
    data = load_summary()
    sizes: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (experiment, config), metrics in data.items():
        if experiment != "scaling" or "_n" not in config:
            continue
        arm, _, n = config.rpartition("_n")
        acc = metrics.get("imputation_accuracy")
        if acc is None:
            continue
        sizes[int(n)][arm] = (acc, metrics.get("imputation_accuracy_std", 0.0))
        # Default explicit baseline in the main sweep is top-8.
        sizes[int(n)]["explicit_top8"] = (
            metrics.get("baseline_explicit_ld_accuracy", np.nan),
            metrics.get("baseline_explicit_ld_accuracy_std", 0.0),
        )
        sizes[int(n)]["majority"] = (
            metrics.get("baseline_majority_accuracy", np.nan),
            metrics.get("baseline_majority_accuracy_std", 0.0),
        )
    strong = load_strong_baseline()
    rate = eval_rate()
    for n, arms in sizes.items():
        entry = strong.get(("scaling", f"both_n{n}", rate))
        if entry:
            # Saturated all-partner control (top_k = L-1).
            arms["explicit_all"] = (entry["accuracy"], entry.get("accuracy_std", 0.0))
    return sorted(sizes.items())


def load_sweep() -> dict[float, dict[str, tuple[float, float]]]:
    path = RESULTS / "mask_rate_sweep.csv"
    if not path.exists():
        return {}
    acc: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["experiment"] != "ablation":
                continue
            rate = float(r["mask_rate"])
            if r["config"] == "both":
                acc[rate]["model"].append(float(r["model"]))
                acc[rate]["explicit_top8"].append(float(r["explicit_ld"]))
                acc[rate]["majority"].append(float(r["majority"]))
            elif r["config"] == "no_bias":
                acc[rate]["ldattention"].append(float(r["model"]))
    out = {
        rate: {s: (float(np.mean(v)), float(np.std(v))) for s, v in series.items()}
        for rate, series in sorted(acc.items())
    }
    strong = load_strong_baseline()
    for (experiment, config, rate), entry in strong.items():
        if experiment == "ablation" and config == "both" and rate in out:
            out[rate]["explicit_all"] = (entry["accuracy"], entry.get("accuracy_std", 0.0))
    return out


def _annotate_bars(ax, bars, values, stds, dy=0.012):
    for bar, val, sd in zip(bars, values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + dy,
            f"{100 * val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def fig_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=11, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)

    ax.text(5, 3.78, "LDAttention", ha="center", fontsize=16, fontweight="bold")
    ax.text(5, 3.50, "Optional LDAttentionBias layer — no explicit $r^2$ matrix",
            ha="center", fontsize=11, color="#555555")

    box(0.25, 2.95, 9.5, 0.40, "genotypes + genomic positions", "#FFF6D6", GOLD)
    box(0.25, 2.32, 9.5, 0.42, "token embedding  →  Q, K, V", "#FFFFFF", BLUE)
    box(0.25, 1.42, 4.55, 0.70, "$B_{distance}$\nper-head log-distance buckets", "#E4F0E6", GREEN, 10)
    box(5.20, 1.42, 4.55, 0.70, "$B_{genotype}$\nsymmetric low-rank bilinear term", "#E4F0E6", GREEN, 10)
    box(0.25, 0.62, 9.5, 0.55,
        r"attention logits  =  $QK^{\top}/\sqrt{d}$  +  $B_{distance}$  +  $B_{genotype}$",
        "#FFF1C2", GOLD, 12, bold=True)
    box(0.25, 0.08, 9.5, 0.38, "softmax  →  attended variants  →  task head", "#FFFFFF", BLUE)

    for x1, y1, x2, y2 in [
        (5, 2.95, 5, 2.74), (2.5, 2.32, 2.5, 2.12), (7.5, 2.32, 7.5, 2.12),
        (2.5, 1.42, 2.5, 1.17), (7.5, 1.42, 7.5, 1.17), (5, 0.62, 5, 0.46),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=DARK))

    out = OUT / "architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_head_to_head() -> Path:
    """Five-way comparison: Majority, Explicit top-8, Explicit all/top-64, LDAttention, +Bias."""
    data = load_summary()
    both = data[("ablation", "both")]
    no_bias = data[("ablation", "no_bias")]
    strong = load_strong_baseline().get(("ablation", "both", eval_rate()), {})
    scaling = dict(load_scaling())
    small_n = min(scaling) if scaling else None
    small = scaling.get(small_n, {}) if small_n else {}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # Shared bar order and colours for both panels.
    # Primary cohort: stronger explicit control is saturated all-partners (top_k=127).
    # Colour role matches window figures where gold = Explicit LD (top-64).
    def _bars(ax, labels, means, stds, colors, title):
        bars = ax.bar(labels, means, yerr=stds, capsize=3.5, color=colors,
                      edgecolor="white", width=0.72)
        _annotate_bars(ax, bars, means, stds, dy=0.010)
        ax.set_ylim(0.55, 1.10)
        ax.set_ylabel("Held-out genotype accuracy")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=9)

    # Left: n=1000 primary
    ax = axes[0]
    labels = [
        "Majority\ngenotype",
        "Explicit LD\n(top-8)",
        "Explicit LD\n(all partners)",
        "LDAttention",
        "LDAttention\n+Bias",
    ]
    means = [
        both["baseline_majority_accuracy"],
        both["baseline_explicit_ld_accuracy"],
        strong.get("accuracy", np.nan),
        no_bias["imputation_accuracy"],
        both["imputation_accuracy"],
    ]
    stds = [
        both["baseline_majority_accuracy_std"],
        both["baseline_explicit_ld_accuracy_std"],
        strong.get("accuracy_std", 0.0),
        no_bias["imputation_accuracy_std"],
        both["imputation_accuracy_std"],
    ]
    colors = [MAJORITY, GOLD_LIGHT, GOLD, SLATE, BLUE]
    _bars(ax, labels, means, stds, colors, "A.  Edge over explicit LD  (1,000 people)")

    # Right: smallest scaling cohort
    ax = axes[1]
    if small:
        labels = [
            "Majority\ngenotype",
            "Explicit LD\n(top-8)",
            "Explicit LD\n(all partners)",
            "LDAttention",
            "LDAttention\n+Bias",
        ]
        means = [
            small.get("majority", (np.nan, 0))[0],
            small.get("explicit_top8", (np.nan, 0))[0],
            small.get("explicit_all", (np.nan, 0))[0],
            small.get("no_bias", (np.nan, 0))[0],
            small.get("both", (np.nan, 0))[0],
        ]
        stds = [
            small.get("majority", (np.nan, 0))[1],
            small.get("explicit_top8", (np.nan, 0))[1],
            small.get("explicit_all", (np.nan, 0))[1],
            small.get("no_bias", (np.nan, 0))[1],
            small.get("both", (np.nan, 0))[1],
        ]
        _bars(ax, labels, means, stds, colors, f"B.  Sample efficiency  ({small_n} people)")
    else:
        ax.axis("off")

    fig.tight_layout()
    out = OUT / "head_to_head.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_robustness() -> Path:
    data = load_summary()
    both = data[("ablation", "both")]
    no_bias = data[("ablation", "no_bias")]
    sweep = load_sweep()
    strong = load_strong_baseline().get(("ablation", "both", eval_rate()), {})

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    if sweep:
        rates = sorted(sweep)
        for key, label, color, marker in (
            ("model", "LDAttention+Bias", BLUE, "o"),
            ("ldattention", "LDAttention", SLATE, "s"),
            ("explicit_all", "Explicit LD (all partners)", GOLD, "^"),
            ("explicit_top8", "Explicit LD (top-8)", GOLD_LIGHT, "v"),
            ("majority", "Majority genotype", MAJORITY, "D"),
        ):
            if key not in sweep[rates[0]]:
                continue
            m = np.array([sweep[r][key][0] for r in rates])
            s = np.array([sweep[r][key][1] for r in rates])
            ax.plot(rates, m, marker=marker, color=color, lw=2.2, ms=7, label=label)
            ax.fill_between(rates, m - s, m + s, color=color, alpha=0.14, lw=0)
        ax.set_xlabel("Fraction of genotypes hidden")
        ax.set_ylabel("Held-out genotype accuracy")
        ax.set_title("A.  Holds as missingness increases")
        ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", fontsize=8.5)
        ax.set_ylim(0.55, 1.02)

    ax = axes[1]
    bins = ["low", "mid", "high"]
    bin_labels = ["MAF 5–10%", "MAF 10–25%", "MAF >25%"]
    width = 0.16
    x = np.arange(len(bins))
    series = (
        ("LDAttention+Bias", BLUE, [both.get(f"accuracy_maf_{b}", np.nan) for b in bins],
         [both.get(f"accuracy_maf_{b}_std", 0.0) for b in bins]),
        ("LDAttention", SLATE, [no_bias.get(f"accuracy_maf_{b}", np.nan) for b in bins],
         [no_bias.get(f"accuracy_maf_{b}_std", 0.0) for b in bins]),
        ("Explicit LD (all partners)", GOLD,
         [strong.get(f"accuracy_maf_{b}", np.nan) for b in bins],
         [strong.get(f"accuracy_maf_{b}_std", 0.0) for b in bins]),
        ("Explicit LD (top-8)", GOLD_LIGHT,
         [both.get(f"baseline_explicit_ld_accuracy_maf_{b}", np.nan) for b in bins],
         [both.get(f"baseline_explicit_ld_accuracy_maf_{b}_std", 0.0) for b in bins]),
        ("Majority", MAJORITY,
         [both.get(f"baseline_majority_accuracy_maf_{b}", np.nan) for b in bins],
         [both.get(f"baseline_majority_accuracy_maf_{b}_std", 0.0) for b in bins]),
    )
    for i, (label, color, vals, errs) in enumerate(series):
        ax.bar(x + (i - 2) * width, vals, width, yerr=errs, capsize=2.5, color=color,
               edgecolor="white", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Held-out genotype accuracy")
    ax.set_title("B.  Rare and common variants")
    ax.set_ylim(0.40, 1.12)
    ax.legend(frameon=True, fancybox=False, edgecolor="#dddddd", fontsize=8.0, loc="lower left")

    fig.tight_layout()
    out = OUT / "robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_attention_vs_ld() -> Path:
    data = load_summary()
    both = data[("ablation", "both")]
    r2 = np.load(RESULTS / "arr_ablation_both_r2.npy")
    attn = np.load(RESULTS / "arr_ablation_both_attn.npy")
    attn_n = (attn - attn.min()) / (attn.max() - attn.min() + 1e-12)
    r = both["attention_vs_r2_pearson"]
    rs = both["attention_vs_r2_pearson_std"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    axes[0].imshow(r2, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("True pairwise $r^2$")
    axes[1].imshow(attn_n, cmap="viridis")
    axes[1].set_title("Learned attention")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Attention recovers LD structure   (Pearson $r$ = {r:.2f} ± {rs:.2f}; $r^2$ was never an input)",
        fontsize=12,
        fontweight="bold",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = OUT / "attention_vs_ld.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_all() -> dict[str, Path]:
    return {
        "architecture": fig_architecture(),
        "head_to_head": fig_head_to_head(),
        "robustness": fig_robustness(),
        "attention_vs_ld": fig_attention_vs_ld(),
    }


if __name__ == "__main__":
    for name, path in make_all().items():
        print(f"{name}: {path}")
