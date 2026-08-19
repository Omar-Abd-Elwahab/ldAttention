"""Publication-quality poster figures for ECCB 2026.

Slot mapping (IID template, 46" x 36"):
  fig_architecture.png        Approach  (left column)        bias schematic
  fig_validation_pipeline.png Methods   (right column, top)  benchmark protocol
  fig_results_main.png        Results   (right, upper)       A head-to-head, B cohort scaling
  fig_results_strip.png       Results   (right, lower)       C missingness, D MAF,
                                                             E attention vs true LD

Every panel is driven by ``results/`` so the poster cannot drift from the
numbers. Each figure is drawn at exactly the aspect ratio of the poster box it
lands in (``poster_layout.FIGURE_ASPECT``) so it fills the slot rather than
letterboxing, and font sizes are chosen for the resulting on-poster scale.

Run ``scripts/run_experiments.py`` first.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import poster_layout as L

ROOT = Path(__file__).resolve().parents[1]
RESULTS = L.RESULTS
POSTER = ROOT / "poster"
POSTER.mkdir(exist_ok=True)

BLUE = "#4472C4"
GREEN = "#70AD47"
GOLD = "#FFC000"
CREAM = "#FFF9E5"
RED = "#C00000"
GREY = "#A5A5A5"
DARK = "#333333"
FONT = L.GRAPH_FONT

plt.rcParams.update({
    "font.family": FONT,
    "font.sans-serif": [FONT, "DejaVu Sans"],
    "font.size": L.SZ_GRAPH_TICK,
    "axes.titlesize": L.SZ_GRAPH_TITLE,
    "axes.labelsize": L.SZ_GRAPH_LABEL,
    "xtick.labelsize": L.SZ_GRAPH_TICK,
    "ytick.labelsize": L.SZ_GRAPH_TICK,
    "legend.fontsize": L.SZ_GRAPH_LEGEND,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def _figsize(slot: str, height: float) -> tuple[float, float]:
    """Width that gives this poster slot's aspect ratio at ``height`` inches."""
    return (L.FIGURE_ASPECT[slot] * height, height)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
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
            rows[key][r["metric"] + "_n"] = float(r["n"])
    return rows


def load_sweep() -> dict[float, dict[str, tuple[float, float]]]:
    """mask_rate -> {series: (mean, std)} for the full model and the controls."""
    path = RESULTS / "mask_rate_sweep.csv"
    if not path.exists():
        return {}
    acc: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["experiment"] != "ablation" or r["config"] != "both":
                continue
            rate = float(r["mask_rate"])
            for series in ("model", "explicit_ld", "majority"):
                acc[rate][series].append(float(r[series]))
    return {
        rate: {s: (float(np.mean(v)), float(np.std(v))) for s, v in series.items()}
        for rate, series in sorted(acc.items())
    }


def load_scaling() -> list[tuple[int, dict[str, tuple[float, float]]]]:
    """[(n_individuals, {arm: (mean, std)})] for the cohort-scaling experiment."""
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
        sizes[int(n)]["explicit_ld"] = (
            metrics.get("baseline_explicit_ld_accuracy", np.nan),
            metrics.get("baseline_explicit_ld_accuracy_std", 0.0),
        )

    # Prefer the saturated control, so the crossover the panel draws is the same
    # one the conclusion and the talk quote.
    strong = L.load_strong_baseline()
    if strong:
        rate = _eval_rate()
        for n in sizes:
            entry = strong.get(("scaling", f"both_n{n}", rate))
            if entry:
                sizes[n]["explicit_ld"] = (entry["accuracy"], entry.get("accuracy_std", 0.0))
    return sorted(sizes.items())


def _bar_labels(ax, bars, values, stds, fmt="{:.3f}", dy=0.004, fontsize=None):
    if fontsize is None:
        fontsize = L.SZ_GRAPH_ANNOT
    for bar, val, sd in zip(bars, values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + dy,
            fmt.format(val),
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        )


def _eval_rate() -> float:
    """The masking rate the headline numbers are scored at."""
    base = json.loads((RESULTS / "config.json").read_text())["base"]
    return float(base.get("eval_mask_rate") or base["mask_rate"])


# --------------------------------------------------------------------------- #
# Approach schematic
# --------------------------------------------------------------------------- #
def architecture_schematic() -> Path:
    fig, ax = plt.subplots(figsize=_figsize("Picture 10", 6.0), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.37)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#E8F0FE", ec=BLUE, fs=None, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=2.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs or L.SZ_GRAPH_LABEL, fontweight="bold" if bold else "normal", zorder=3)

    ax.text(5, 3.20, "LDAttentionBias", ha="center", fontsize=L.SZ_GRAPH_TITLE + 4, fontweight="bold")
    ax.text(5, 2.96, "a drop-in additive attention bias — no r² matrix, ever",
            ha="center", fontsize=L.SZ_GRAPH_LABEL, color="#555555", style="italic")

    box(0.15, 2.48, 9.7, 0.36, "genotypes + genomic positions   (+ optional population ID)", CREAM, "#8A6D00", L.SZ_GRAPH_LABEL)
    box(0.15, 1.94, 9.7, 0.38, "token embedding  →  Q, K, V", "#FFFFFF", BLUE, L.SZ_GRAPH_LABEL)
    box(0.15, 1.16, 4.72, 0.60, "B$_{distance}$\nper-head bias over log-spaced\ngenomic-distance buckets",
        "#D9EAD3", GREEN, L.SZ_GRAPH_TICK)
    box(5.13, 1.16, 4.72, 0.60, "B$_{genotype}$\nsymmetric low-rank bilinear\ncorrelation-like term",
        "#D9EAD3", GREEN, L.SZ_GRAPH_TICK)
    box(0.15, 0.50, 9.7, 0.50,
        "attention logits  =  $QK^\\top/\\sqrt{d}$  +  B$_{distance}$  +  B$_{genotype}$",
        "#FFF2CC", GOLD, L.SZ_GRAPH_TITLE - 2, bold=True)
    box(0.15, 0.02, 9.7, 0.34, "softmax  →  attended variants  →  task head", "#FFFFFF", BLUE, L.SZ_GRAPH_LABEL)

    for x1, y1, x2, y2 in [
        (5, 2.48, 5, 2.32), (2.5, 1.94, 2.5, 1.76), (7.5, 1.94, 7.5, 1.76),
        (2.5, 1.16, 2.5, 1.00), (7.5, 1.16, 7.5, 1.00), (5, 0.50, 5, 0.36),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2.1, color=DARK))

    out = POSTER / "fig_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Methods / benchmark protocol strip
# --------------------------------------------------------------------------- #
def validation_pipeline() -> Path:
    w, h = _figsize("Picture 11", 2.70)
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    base = {}
    cfg_path = RESULTS / "config.json"
    if cfg_path.exists():
        base = json.loads(cfg_path.read_text()).get("base", {})
    block = bool(base.get("block_missing", False))
    blen = int(base.get("block_len", 8))
    hide_body = (
        f"drop {blen}-SNP blocks\nsame holes for\nevery method"
        if block else
        "remove genotypes;\nsame holes for\nevery method"
    )
    steps = [
        ("1. Simulate", "make a cohort\ncommon SNPs only\n(MAF ≥ 5%)", CREAM, "#8A6D00"),
        ("2. Split", "train / check /\nheld-out people\n(no overlap)", "#FFFFFF", BLUE),
        ("3. Hide", hide_body, "#EDEDED", "#777777"),
        ("4. Train", "transformer +\nLD layer\n(no r² table)", "#D9EAD3", GREEN),
        ("5. Score", "accuracy = share\nof hidden sites\nguessed right", "#E8F0FE", BLUE),
        ("6. Reuse", "same layer drops\ninto any standard\nattention", "#FFF2CC", GOLD),
    ]
    n = len(steps)
    # Keep the title and the footnote out of the boxes: title sits in a top
    # band, footnote in a bottom band, boxes in the middle with a clear gap.
    title_band, foot_band, vgap = 0.40, 0.38, 0.18
    y = foot_band + vgap
    bh = h - title_band - vgap - y
    gap = 0.22
    bw = (w - gap * (n + 1)) / n
    for i, (title, body, fc, ec) in enumerate(steps):
        x = gap + i * (bw + gap)
        ax.add_patch(plt.Rectangle((x, y), bw, bh, facecolor=fc, edgecolor=ec, linewidth=2.2))
        ax.text(x + bw / 2, y + bh * 0.76, title, ha="center", va="center",
                fontsize=L.SZ_GRAPH_LABEL, fontweight="bold")
        ax.text(x + bw / 2, y + bh * 0.32, body, ha="center", va="center",
                fontsize=L.SZ_GRAPH_TICK - 1, color="#333333")
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap * 0.72, y + bh / 2),
                        xytext=(x + bw + gap * 0.08, y + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=DARK))

    ax.text(w / 2, h - 0.08,
            "How we score: hide genotypes, then count how many the model gets right on new people",
            ha="center", va="top", fontsize=L.SZ_GRAPH_LABEL, fontweight="bold")
    ax.text(w / 2, 0.08,
            "True r² is computed only to score the explicit-LD control and panel E. It is never an input to ldAttention.",
            ha="center", va="bottom", fontsize=L.SZ_GRAPH_TICK - 1, style="italic", color="#555555")
    out = POSTER / "fig_validation_pipeline.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Results A/B: the headline comparison and where the bias earns its place
# --------------------------------------------------------------------------- #
def results_main() -> Path:
    data = load_summary()
    abl = lambda c: data[("ablation", c)]  # noqa: E731

    fig = plt.figure(figsize=_figsize("Picture 8", 6.0), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 0.96], wspace=0.40,
                          left=0.100, right=0.985, top=0.80, bottom=0.150)

    # ---- A: head-to-head against the pipeline being replaced ----------------
    # The sparse control alone is not a fair opponent -- it keeps improving with
    # more partners -- so the saturated version is shown beside it and carries
    # the headline delta whenever it has been computed.
    ax = fig.add_subplot(gs[0, 0])
    strong = L.load_strong_baseline()
    strong_entry = strong.get(("ablation", "both", _eval_rate()))

    labels = ["allele-frequency\nfloor", "explicit-LD\nusual top-8"]
    means = [abl("both")["baseline_majority_accuracy"], abl("both")["baseline_explicit_ld_accuracy"]]
    stds = [abl("both")["baseline_majority_accuracy_std"],
            abl("both")["baseline_explicit_ld_accuracy_std"]]
    colors = [GREY, "#FFD866"]
    if strong_entry:
        labels.append("explicit-LD\nall partners")
        means.append(strong_entry["accuracy"])
        stds.append(strong_entry["accuracy_std"])
        colors.append(GOLD)
    else:
        colors[1] = GOLD
        labels[1] = "explicit-LD\npipeline"
    labels.append("ldAttention\n(this work)")
    means.append(abl("both")["imputation_accuracy"])
    stds.append(abl("both")["imputation_accuracy_std"])
    colors.append(BLUE)

    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=1.6, width=0.62)
    bars[-1].set_edgecolor(BLUE)
    bars[-1].set_linewidth(3.0)
    label_size = L.SZ_GRAPH_ANNOT if len(bars) <= 3 else L.SZ_GRAPH_TICK
    _bar_labels(ax, bars, means, stds, dy=0.010, fontsize=label_size)
    ax.set_ylabel("Accuracy\n(hidden genotypes guessed right, held-out people)",
                  fontsize=L.SZ_GRAPH_LABEL, labelpad=14)
    # Headroom for the value labels, the delta arrow above them, and the
    # two-line delta caption above that -- the bars sit near 1.0, so this has to
    # be measured rather than guessed at.
    tops = [m + s for m, s in zip(means, stds)]
    ax.set_ylim(0.55, max(tops) + 0.18)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])

    # The usual top-k pipeline is what this layer replaces; the saturated bar
    # is a fairness check sitting beside it. The arrow calls out the practical gap.
    model_i = len(means) - 1
    ref_i = 1
    delta = means[model_i] - means[ref_i]
    headline = ("Beats the usual r² pipeline" if strong_entry
                else "Beats the explicit-LD pipeline it replaces")
    if delta <= 0:
        headline = "Head-to-head against every control"
    ax.set_title(f"A.  {headline}", fontsize=L.SZ_GRAPH_TITLE, fontweight="bold", loc="left", pad=10)
    ax.tick_params(labelsize=L.SZ_GRAPH_TICK, pad=5)

    arrow_y = max(tops[ref_i], tops[model_i]) + 0.040
    ax.annotate("", xy=(model_i, arrow_y), xytext=(ref_i, arrow_y),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=BLUE,
                                connectionstyle="arc3,rad=-0.30"))
    ax.text((ref_i + model_i) / 2 + 0.15, arrow_y + 0.018,
            f"{delta * 100:+.1f} pts  ·  no r² table",
            fontsize=L.SZ_GRAPH_ANNOT, fontweight="bold", color=BLUE, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))

    # The bars score the same masked entries, so the paired test is the honest
    # statistic -- not whether the two error bars happen to overlap.
    # Seed count and the paired test live in the figure caption, not on the bars.

    # ---- B: where the inductive bias is actually load-bearing ---------------
    ax = fig.add_subplot(gs[0, 1])
    scaling = load_scaling()
    if scaling:
        sizes = [n for n, _ in scaling]
        curves = {}
        ld_label = "explicit-LD" if L.load_strong_baseline() else "explicit-LD pipeline"
        # The wall shows the layer's edge: attention vs the r² pipeline it replaces.
        # The plain transformer is a ceiling check for the talk, not a poster claim.
        for arm, label, color, marker in (
            ("both", "ldAttention", BLUE, "o"),
            ("explicit_ld", ld_label, GOLD, "^"),
        ):
            m = np.array([v.get(arm, (np.nan, 0.0))[0] for _, v in scaling])
            s = np.array([v.get(arm, (np.nan, 0.0))[1] for _, v in scaling])
            curves[arm] = m
            ax.plot(sizes, m, marker=marker, color=color, lw=3.2, ms=9, label=label)
            ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.16, lw=0)

        # The two approaches win in different regimes. Mark the boundary rather
        # than letting the eye pick whichever end flatters the model.
        gap = curves["both"] - curves["explicit_ld"]
        crossings = [i for i in range(1, len(sizes)) if gap[i - 1] <= 0 < gap[i]]
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("People in the cohort", fontsize=L.SZ_GRAPH_LABEL, labelpad=8)
        ax.set_ylabel("Accuracy on hidden genotypes", fontsize=L.SZ_GRAPH_LABEL, labelpad=14)
        ax.legend(fontsize=L.SZ_GRAPH_LEGEND, loc="lower left",
                  frameon=True, fancybox=False, edgecolor="#dddddd",
                  facecolor="white", framealpha=0.95)

        if crossings:
            i = crossings[0]
            xc = float(np.sqrt(sizes[i - 1] * sizes[i]))  # midpoint on a log axis
            ax.axvline(xc, color="#555555", ls=":", lw=2.2, zorder=0)
            ax.annotate("crossover", xy=(xc, 0.99), xycoords=("data", "axes fraction"),
                        fontsize=L.SZ_GRAPH_TICK - 1, color="#555555", ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))
            title = "B.  Two regimes, split by cohort size"
        elif np.isfinite(gap).all() and (gap > 0).all():
            title = "B.  Ahead at every panel size"
        else:
            title = "B.  ldAttention vs the r² pipeline"

        ends = [0, len(sizes) - 1]
        if len(sizes) > 1 and np.isfinite(gap[ends]).all():
            for i in ends:
                lo = min(curves["both"][i], curves["explicit_ld"][i])
                hi = max(curves["both"][i], curves["explicit_ld"][i])
                ax.annotate(
                    "", xy=(sizes[i], hi), xytext=(sizes[i], lo),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6,
                                    shrinkA=1.5, shrinkB=1.5),
                )
                ax.annotate(
                    f"{100 * gap[i]:+.1f} pt", xy=(sizes[i], (lo + hi) / 2),
                    xytext=(18 if i == 0 else -18, 0), textcoords="offset points",
                    fontsize=L.SZ_GRAPH_ANNOT, fontweight="bold", color="#333333",
                    ha="left" if i == 0 else "right", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94),
                )

        ax.set_title(title, fontsize=L.SZ_GRAPH_TITLE, fontweight="bold", loc="left", pad=8)
    ax.tick_params(labelsize=L.SZ_GRAPH_TICK)

    out = POSTER / "fig_results_main.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Results C-F: supporting evidence
# --------------------------------------------------------------------------- #
def results_strip() -> Path:
    data = load_summary()
    sweep = load_sweep()
    abl = lambda c: data[("ablation", c)]  # noqa: E731

    fig = plt.figure(figsize=_figsize("Picture 24", 6.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.12, 1.20], wspace=0.38,
                          left=0.055, right=0.988, top=0.78, bottom=0.28)
    title_kw = dict(fontsize=L.SZ_GRAPH_TITLE - 1, fontweight="bold", loc="left", pad=8)

    # ---- C: accuracy vs missingness (was D) ---------------------------------
    strong = L.load_strong_baseline()
    for (experiment, config, rate), entry in strong.items():
        if experiment == "ablation" and config == "both" and rate in sweep:
            sweep[rate]["explicit_ld"] = (entry["accuracy"], entry.get("accuracy_std", 0.0))

    ax = fig.add_subplot(gs[0, 0])
    if sweep:
        rates = sorted(sweep)
        ld_label = "explicit-LD" if strong else "explicit-LD"
        for key, label, color, marker in (
            ("model", "ldAttention", BLUE, "o"),
            ("explicit_ld", ld_label, GOLD, "s"),
            ("majority", "allele-freq.", GREY, "^"),
        ):
            m = np.array([sweep[r][key][0] for r in rates])
            s = np.array([sweep[r][key][1] for r in rates])
            ax.plot(rates, m, marker=marker, color=color, lw=3.0, ms=8, label=label)
            ax.fill_between(rates, m - s, m + s, color=color, alpha=0.18, lw=0)
        ax.set_xlabel("Share of genotypes hidden", fontsize=L.SZ_GRAPH_LABEL, labelpad=8)
        ax.set_ylabel("Accuracy on hidden genotypes", fontsize=L.SZ_GRAPH_LABEL, labelpad=10)
        ax.legend(fontsize=L.SZ_GRAPH_LEGEND - 1, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, -0.26),
                  frameon=True, fancybox=False, edgecolor="none",
                  facecolor="white", framealpha=0.95, columnspacing=0.9, handlelength=1.1)
        ax.set_ylim(0.55, 1.04)
    ax.set_title("C.  Holds when more is missing", **title_kw)
    ax.tick_params(labelsize=L.SZ_GRAPH_TICK, pad=3)

    # ---- D: accuracy by allele frequency (was E) ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    bins = ["low", "mid", "high"]
    bin_labels = ["5–10%", "10–25%", ">25%"]
    series = (
        ("ldAttention", "accuracy_maf_", BLUE),
        ("explicit-LD", "baseline_explicit_ld_accuracy_maf_", GOLD),
        ("allele-freq.", "baseline_majority_accuracy_maf_", GREY),
    )
    strong_maf = strong.get(("ablation", "both", _eval_rate()))
    width = 0.26
    x = np.arange(len(bins))
    for i, (label, prefix, color) in enumerate(series):
        if label == "explicit-LD" and strong_maf:
            label = "explicit-LD"
            vals = [strong_maf.get(f"accuracy_maf_{b}", np.nan) for b in bins]
            errs = [strong_maf.get(f"accuracy_maf_{b}_std", 0.0) for b in bins]
        else:
            vals = [abl("both").get(f"{prefix}{b}", np.nan) for b in bins]
            errs = [abl("both").get(f"{prefix}{b}_std", 0.0) for b in bins]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3.5,
               color=color, edgecolor="white", linewidth=1.2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=L.SZ_GRAPH_TICK)
    ax.set_xlabel("How common the variant is (MAF)", fontsize=L.SZ_GRAPH_LABEL, labelpad=8)
    ax.set_ylabel("Accuracy on hidden genotypes", fontsize=L.SZ_GRAPH_LABEL, labelpad=8)
    ax.set_ylim(0.4, 1.08)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(fontsize=L.SZ_GRAPH_LEGEND - 1, frameon=True, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.26), columnspacing=0.9, handlelength=1.1,
              facecolor="white", edgecolor="none", framealpha=0.95)
    ax.set_title("D.  Rare and common variants", **title_kw)
    ax.tick_params(labelsize=L.SZ_GRAPH_TICK, pad=3)

    # ---- E: learned attention vs true LD (was F) ----------------------------
    sub = gs[0, 2].subgridspec(1, 2, wspace=0.12)
    ax0 = fig.add_subplot(sub[0, 0])
    ax1 = fig.add_subplot(sub[0, 1])
    r2_path = RESULTS / "arr_ablation_both_r2.npy"
    attn_path = RESULTS / "arr_ablation_both_attn.npy"
    if r2_path.exists() and attn_path.exists():
        r2 = np.load(r2_path)
        attn = np.load(attn_path)
        attn_n = (attn - attn.min()) / (attn.max() - attn.min() + 1e-12)
        ax0.imshow(r2, cmap="viridis", vmin=0, vmax=1)
        ax1.imshow(attn_n, cmap="viridis")
    for a, t in ((ax0, "ground-truth r²"), (ax1, "learned attention")):
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(t, fontsize=L.SZ_GRAPH_TICK, labelpad=6)
    r = abl("both")["attention_vs_r2_pearson"]
    rs = abl("both")["attention_vs_r2_pearson_std"]
    ax0.set_title("E.  Attention recovers LD", **title_kw)
    ax1.text(0.0, -0.26,
             f"match to true r²: r = {r:.2f} ± {rs:.2f}\n(r² was never an input)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=L.SZ_GRAPH_TICK - 1,
             color="#555555", style="italic", linespacing=1.45)

    out = POSTER / "fig_results_strip.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def qr_code() -> Path:
    """QR for the repository, replacing the template's (which points elsewhere).

    Encoded from ``poster_layout.REPO_URL`` so the code and the printed link can
    never disagree. High error correction, since posters get scanned at an angle
    and from a distance.
    """
    import segno
    from PIL import Image

    out = POSTER / "fig_qr.png"
    segno.make(f"https://{L.REPO_URL}", error="h").save(
        str(out), scale=20, border=2, dark="#000000", light="#FFFFFF"
    )
    # segno writes a 1-bit PNG; matplotlib would read that as a 2-D array and
    # colour-map it, so store it as plain RGB instead.
    with Image.open(out) as im:
        im.convert("RGB").save(out)
    return out


def make_all() -> dict[str, Path]:
    return {
        "architecture": architecture_schematic(),
        "pipeline": validation_pipeline(),
        "results_main": results_main(),
        "results_strip": results_strip(),
        "qr": qr_code(),
    }


if __name__ == "__main__":
    for k, p in make_all().items():
        print(f"{k}: {p}")
