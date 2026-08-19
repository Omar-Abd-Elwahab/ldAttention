"""Single source of truth for the ECCB 2026 poster: geometry and copy.

Both outputs are driven from here so they cannot drift apart:

* ``scripts/build_poster.py``      -> editable .pptx (keeps the IID template's
  real fonts, logos and footer, repositioned onto the grid below)
* ``scripts/render_poster_pdf.py`` -> the print PDF, composed directly (there is
  no LibreOffice on this machine to convert the .pptx)

Geometry is in inches on the template's 46 x 36 in canvas. The design language is
the IID template's: numbered circular section badges, a full-height vertical rule
splitting two columns, and a cream footer strip carrying the author block, the
partner logos and the QR code.

Two defects in the previous build are fixed here by construction:

* section labels used to sit *under* their circle in a box too narrow for the
  word, so they wrapped mid-word ("Backgrou nd"). They now sit *beside* the
  circle on a line wide enough for the longest label.
* the sections used to be numbered 1, 2, 3, 3, 4. They are now 1-5.

All headline numbers come from ``results/`` at build time -- nothing is hardcoded.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.stats import wilcoxon_signed_rank  # noqa: E402
# Overridable so a build can be rehearsed against a scratch sweep. Prefer the
# larger block-missing run when it exists; that is what the poster is built from.
_DEFAULT_RESULTS = ROOT / "results_large"
if not (_DEFAULT_RESULTS / "results.json").exists():
    _DEFAULT_RESULTS = ROOT / "results"
RESULTS = Path(os.environ.get("LDATTENTION_RESULTS", _DEFAULT_RESULTS))
SMALL = Path(os.environ.get("LDATTENTION_RESULTS_SMALL", ROOT / "results_small"))
POSTER = ROOT / "poster"
TEMPLATE = Path("/media/omar/Projects/ULAVAL/Omar_Abdelwahab_IID.pptx")

# --------------------------------------------------------------------------- #
# Canvas
# --------------------------------------------------------------------------- #
PAGE_W, PAGE_H = 46.0, 36.0
FOOTER_TOP = 32.03  # cream strip in the template; content must stay above it

LEFT_X0, LEFT_X1 = 0.55, 22.40
RIGHT_X0, RIGHT_X1 = 23.45, 45.55
LEFT_W = LEFT_X1 - LEFT_X0
RIGHT_W = RIGHT_X1 - RIGHT_X0

DIVIDER_X = 22.97
DIVIDER_Y0, DIVIDER_Y1 = 1.18, 31.60

# --------------------------------------------------------------------------- #
# Palette (template accent colours)
# --------------------------------------------------------------------------- #
NAVY = "#1F3864"
BLUE = "#4472C4"
GREEN = "#70AD47"
GOLD = "#FFC000"
RED = "#C00000"
GREY = "#A5A5A5"
CREAM = "#FFF9E5"
INK = "#1A1A1A"
MUTED = "#555555"

# Fonts. The .pptx asks for the template faces (Baskerville / Abadi). Those
# files are not on this machine, so the print PDF and every graph use the same
# installed pair — Liberation Serif for titles, Liberation Sans for body and
# figures — so the wall and the charts cannot disagree.
PPTX_FONT_TITLE = "Baskerville Old Face"
PPTX_FONT_BODY = "Abadi"
PDF_FONT_TITLE = ["Liberation Serif", "DejaVu Serif", "serif"]
PDF_FONT_BODY = ["Liberation Sans", "DejaVu Sans", "sans-serif"]
GRAPH_FONT = "Liberation Sans"


@dataclass(frozen=True)
class Box:
    """A rectangle in inches, measured from the top-left of the page."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
TITLE = Box(LEFT_X0, 0.45, LEFT_W, 4.55)
TITLE_RULE_Y = 5.30

# Left column -- Background, Approach
BG_HEADER = Box(LEFT_X0, 5.70, LEFT_W, 1.75)
BG_BODY = Box(LEFT_X0, 7.60, LEFT_W, 6.10)
AP_HEADER = Box(LEFT_X0, 14.05, LEFT_W, 1.75)
AP_FIGURE = Box(LEFT_X0, 15.95, LEFT_W, 7.15)
AP_BODY = Box(LEFT_X0, 23.50, LEFT_W, 8.00)

# Right column -- Methods, Results, Conclusion
ME_HEADER = Box(RIGHT_X0, 0.45, RIGHT_W, 1.65)
ME_FIGURE = Box(RIGHT_X0, 2.20, RIGHT_W, 2.65)
ME_NOTE = Box(RIGHT_X0, 5.25, RIGHT_W, 1.88)
RE_HEADER = Box(RIGHT_X0, 7.32, RIGHT_W, 1.65)
RE_FIGURE_A = Box(RIGHT_X0, 9.15, RIGHT_W, 7.45)
RE_FIGURE_B = Box(RIGHT_X0, 16.80, RIGHT_W, 5.80)
RE_CAPTION = Box(RIGHT_X0, 23.55, RIGHT_W, 1.38)
CO_HEADER = Box(RIGHT_X0, 25.05, RIGHT_W, 1.70)
# The conclusion is where the eye lands after the figures, so it carries three
# scannable takeaways rather than a paragraph, with the scope caveat kept below
# them in smaller type instead of being buried mid-sentence.
CO_BODY = Box(RIGHT_X0, 26.90, RIGHT_W, 2.70)
CO_NOTE = Box(RIGHT_X0, 29.75, RIGHT_W, 1.25)

# Badge geometry: circle then label, side by side, on one line.
BADGE_DIAMETER = 1.35
BADGE_GAP = 0.45

SECTIONS = (
    ("1", "Background", BG_HEADER),
    ("2", "Approach", AP_HEADER),
    ("3", "Methods", ME_HEADER),
    ("4", "Results", RE_HEADER),
    ("5", "Conclusion", CO_HEADER),
)

# Figure slot -> generated file. Slot names are the template picture shapes the
# .pptx builder swaps out.
FIGURES = {
    "Picture 10": ("fig_architecture.png", AP_FIGURE),         # Approach
    "Picture 11": ("fig_validation_pipeline.png", ME_FIGURE),  # Methods
    "Picture 8": ("fig_results_main.png", RE_FIGURE_A),        # Results A/B
    "Picture 24": ("fig_results_strip.png", RE_FIGURE_B),      # Results C-F
}

# Each generated figure is drawn at exactly its slot's aspect ratio, so it fills
# the box instead of letterboxing into it.
FIGURE_ASPECT = {name: box.w / box.h for name, (_, box) in FIGURES.items()}

# --------------------------------------------------------------------------- #
# Type scale (points, at 46 x 36 in)
# --------------------------------------------------------------------------- #
SZ_TITLE = 86
SZ_SUBTITLE = 42
SZ_BADGE_NUM = 56
SZ_BADGE_LABEL = 58
SZ_BODY = 36
SZ_BULLET = 32
SZ_CAPTION = 26
SZ_FOOTER = 26
SZ_FOOTER_SMALL = 22
# Graphs are read at arm's length on a 46 × 36 in board. These sizes are the
# on-figure point sizes; the PNG is then scaled up into its slot (~1.25× for
# the results panels), so 22 pt on disk lands near 28 pt on the wall.
SZ_GRAPH_TITLE = 22
SZ_GRAPH_LABEL = 18
SZ_GRAPH_TICK = 16
SZ_GRAPH_LEGEND = 15
SZ_GRAPH_ANNOT = 16

TITLE_MAIN = "LD-Aware Attention for Genomic Analysis"
TITLE_SUB = "Fill in missing genotypes without building an r² table"

REPO_URL = "github.com/omar-abdelwahab/ldAttention"


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Headline:
    """Every number quoted in the poster copy, read from ``results/``."""

    model: float
    model_std: float
    explicit: float
    explicit_std: float
    majority: float
    majority_std: float
    delta: float
    delta_std: float
    wins: int
    n_seeds: int
    dosage_r2: float
    attn_r2: float
    attn_r2_std: float
    n_sites: int
    n_individuals: int
    mask_rate: float
    pop_gain: float
    pop_base: float
    pop_film: float
    plain: float
    small_model: float
    small_plain: float
    small_n: int
    # The 4-arm ablation actually drawn as panel C, which may be a different
    # cohort than small_n (the sample-efficiency claim). The caption has to
    # describe the chart, not the talk.
    ablation_n: int
    ablation_model: float
    ablation_plain: float
    ablation_seeds: int
    scaling_seeds: int
    p_value: float
    crossover_n: int | None
    # Attention↔true-r² of the plain transformer, so the talk can name the
    # layer's edge when accuracy has already saturated.
    attn_plain: float
    block_missing: bool
    block_len: int
    # The saturated explicit-LD control. NaN until strong_baseline_pass.py runs.
    strong: float
    strong_std: float
    strong_delta: float
    strong_p: float
    strong_top_k: int

    def pct(self, value: float) -> str:
        return f"{100 * value:.1f}%"

    @property
    def has_strong(self) -> bool:
        return self.strong == self.strong  # False only for NaN

    @property
    def p_text(self) -> str:
        """Exact two-sided Wilcoxon signed-rank on the per-seed paired deltas."""
        return self._fmt_p(self.p_value)

    @property
    def strong_p_text(self) -> str:
        return self._fmt_p(self.strong_p)

    @staticmethod
    def _fmt_p(p: float) -> str:
        if p != p:
            return "p n/a"
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _summary_lookup() -> dict[tuple[str, str, str], dict[str, float]]:
    import csv

    table: dict[tuple[str, str, str], dict[str, float]] = {}
    with (RESULTS / "results_summary.csv").open() as f:
        for row in csv.DictReader(f):
            table[(row["experiment"], row["config"], row["metric"])] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "sem": float(row.get("sem") or 0.0),
                "n": float(row["n"]),
            }
    return table


def _paired_values(experiment: str, config: str, metric: str) -> dict[int, float]:
    """``{seed: value}`` for one metric of one arm, from the raw per-run table."""
    import csv

    raw = RESULTS / "results_raw.csv"
    if not raw.exists():
        return {}
    out: dict[int, float] = {}
    with raw.open() as f:
        for row in csv.DictReader(f):
            if row["experiment"] != experiment or row["config"] != config:
                continue
            value = row.get(metric)
            if value in (None, "", "nan"):
                continue
            out[int(row["seed"])] = float(value)
    return out


def _paired_deltas(experiment: str, config: str, metric: str) -> list[float]:
    """Per-seed values of a paired difference metric, in seed order."""
    values = _paired_values(experiment, config, metric)
    return [v for _, v in sorted(values.items())]


def _ablation_arms(directory: Path) -> tuple[int, float, float]:
    """``(n_individuals, both, no_bias)`` from one sweep's summary, or zeros."""
    import csv

    summary = directory / "results_summary.csv"
    config = directory / "config.json"
    if not summary.exists() or not config.exists():
        return 0, float("nan"), float("nan")

    arms = {"both": float("nan"), "no_bias": float("nan")}
    with summary.open() as f:
        for row in csv.DictReader(f):
            if row["experiment"] == "ablation" and row["metric"] == "imputation_accuracy":
                if row["config"] in arms:
                    arms[row["config"]] = float(row["mean"])
    n = json.loads(config.read_text())["base"]["n_haplotypes"] // 2
    return n, arms["both"], arms["no_bias"]


def _protocol(directory: Path) -> tuple:
    """The simulation choices that have to match before two sweeps are comparable."""
    cfg = json.loads((directory / "config.json").read_text())["base"]
    return (
        int(cfg["n_sites"]),
        bool(cfg.get("block_missing", False)),
        int(cfg.get("block_len", 8)),
        round(float(cfg.get("min_maf", 0.05)), 4),
        bool(cfg.get("use_msprime", False)),
    )


def compatible_small_dir() -> Path | None:
    """The small-cohort 4-arm sweep, or None if it is a different experiment.

    An older 64-site iid-mask run must not be drawn next to a 128-site
    block-missing headline; the caption would then describe the wrong chart.
    """
    if not (SMALL / "config.json").exists() or not (SMALL / "results_summary.csv").exists():
        return None
    if not (RESULTS / "config.json").exists():
        return SMALL
    try:
        if _protocol(SMALL) != _protocol(RESULTS):
            return None
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    return SMALL


def _small_cohort_arms(scaling: dict) -> tuple[int, float, float]:
    """Smallest cohort at which both vs no-bias was measured.

    Prefers the dedicated small sweep only when it is actually smaller than the
    scaling curve; a 200-individual 4-arm run must not hide a 100-individual
    scaling point, which is where the sample-efficiency claim lives.
    """
    candidates: list[tuple[int, float, float]] = []
    small_dir = compatible_small_dir()
    small = _ablation_arms(small_dir) if small_dir else (0, float("nan"), float("nan"))
    if small[0]:
        candidates.append(small)
    sizes = sorted(
        int(name.split("_n")[-1]) for name in scaling if name.startswith("both_n")
    )
    if sizes:
        n = sizes[0]
        candidates.append((
            n,
            scaling.get(f"both_n{n}", {}).get("imputation_accuracy", {}).get("mean", float("nan")),
            scaling.get(f"no_bias_n{n}", {}).get("imputation_accuracy", {}).get("mean", float("nan")),
        ))
    if not candidates:
        return 0, float("nan"), float("nan")
    return min(candidates, key=lambda row: row[0])


def load_strong_baseline() -> dict[tuple[str, str, float], dict]:
    """The saturated explicit-LD control, keyed by (experiment, config, mask rate).

    Written by ``scripts/strong_baseline_pass.py``. Absent until that pass has
    been run, in which case the poster falls back to reporting the sparse control
    alone -- so every caller must handle the empty dict.
    """
    import csv
    from collections import defaultdict

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
            grouped[key]["seed"].append(int(row["seed"]))
            meta[key] = {"top_k": int(row["top_k"]), "epochs": int(row["epochs"])}

    out: dict[tuple[str, str, float], dict] = {}
    for key, fields in grouped.items():
        entry: dict = dict(meta[key])
        # Sorted, to line up with the *_values lists below.
        entry["seeds"] = sorted(fields["seed"])
        for field, values in fields.items():
            if field == "seed" or not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            entry[field] = mean
            entry[f"{field}_std"] = var ** 0.5
            entry[f"{field}_values"] = [v for _, v in sorted(zip(fields["seed"], values))]
        entry["n"] = len(fields["seed"])
        out[key] = entry
    return out


def _crossover(scaling: dict, eval_rate: float) -> int | None:
    """Smallest cohort at which ldAttention overtakes the explicit-LD control.

    The two approaches win in different regimes; the poster has to say where the
    boundary is rather than quoting whichever side flatters the model. Measured
    against the saturated control when it exists, since a crossover against a
    deliberately weak opponent would not mean anything.
    """
    strong = load_strong_baseline()
    points = []
    for name, metrics in scaling.items():
        if not name.startswith("both_n"):
            continue
        n = int(name.split("_n")[-1])
        model = metrics.get("imputation_accuracy", {}).get("mean")
        entry = strong.get(("scaling", name, eval_rate))
        explicit = (
            entry["accuracy"] if entry
            else metrics.get("baseline_explicit_ld_accuracy", {}).get("mean")
        )
        if model is None or explicit is None:
            continue
        points.append((n, model - explicit))
    points.sort()
    if not points or points[0][1] > 0:
        return None  # model already ahead at the smallest cohort tested
    for n, gap in points:
        if gap > 0:
            return n
    return None


def load_headline() -> Headline:
    table = _summary_lookup()
    results = json.loads((RESULTS / "results.json").read_text())
    base = results["config"]["base"]
    summary = results["ablation"]["summary"]["both"]

    def stat(metric: str, experiment: str = "ablation", config: str = "both") -> dict[str, float]:
        return table.get((experiment, config, metric), {"mean": float("nan"), "std": 0.0, "n": 0})

    wins = summary.get("wins_vs_explicit_ld", {})
    pop_base = stat("imputation_accuracy", "population", "both")["mean"]
    pop_film = stat("imputation_accuracy", "population", "both_pop_film")["mean"]

    n_haps = base["n_haplotypes"]
    scaling = results.get("scaling", {}).get("summary", {})
    small_n, small_model, small_plain = _small_cohort_arms(scaling)
    # Panel C is the 4-arm ablation. A dedicated small sweep is what that panel
    # is for; without it the chart is the main-cohort ablation and will read as
    # saturated, which the title and caption both have to admit. Ignore a small
    # sweep from a different protocol (old 64-site iid masks vs 128-site blocks).
    small_dir = compatible_small_dir()
    ablation_n, ablation_model, ablation_plain = (
        _ablation_arms(small_dir) if small_dir else (0, float("nan"), float("nan"))
    )
    ablation_seeds = 0
    if ablation_n and small_dir is not None:
        small_cfg = json.loads((small_dir / "config.json").read_text())
        ablation_seeds = len(small_cfg.get("seeds") or [])
    else:
        ablation_n = n_haps // 2
        ablation_model = stat("imputation_accuracy")["mean"]
        ablation_plain = stat("imputation_accuracy", "ablation", "no_bias")["mean"]
        ablation_seeds = len(results["config"]["seeds"])

    deltas = _paired_deltas("ablation", "both", "model_minus_explicit_ld")
    paired = wilcoxon_signed_rank(deltas) if deltas else None

    # Against the saturated control the margin is smaller and the pairing has to
    # be rebuilt by seed, since the two passes wrote separate files.
    eval_rate = base.get("eval_mask_rate") or base["mask_rate"]
    strong_entry = load_strong_baseline().get(("ablation", "both", eval_rate), {})
    model_by_seed = _paired_values("ablation", "both", "imputation_accuracy")
    strong_deltas: list[float] = []
    if strong_entry:
        for seed, value in zip(strong_entry["seeds"], strong_entry["accuracy_values"]):
            if seed in model_by_seed:
                strong_deltas.append(model_by_seed[seed] - value)
    strong_paired = wilcoxon_signed_rank(strong_deltas) if strong_deltas else None

    scaling_seeds = results["config"].get("scaling_seeds") or []
    return Headline(
        model=stat("imputation_accuracy")["mean"],
        model_std=stat("imputation_accuracy")["std"],
        explicit=stat("baseline_explicit_ld_accuracy")["mean"],
        explicit_std=stat("baseline_explicit_ld_accuracy")["std"],
        majority=stat("baseline_majority_accuracy")["mean"],
        majority_std=stat("baseline_majority_accuracy")["std"],
        delta=stat("model_minus_explicit_ld")["mean"],
        delta_std=stat("model_minus_explicit_ld")["std"],
        wins=paired.n_wins if paired else int(wins.get("n_wins", 0)),
        n_seeds=len(results["config"]["seeds"]),
        dosage_r2=stat("dosage_r2")["mean"],
        attn_r2=stat("attention_vs_r2_pearson")["mean"],
        attn_r2_std=stat("attention_vs_r2_pearson")["std"],
        n_sites=base["n_sites"],
        n_individuals=n_haps // 2,
        mask_rate=base["mask_rate"],
        pop_gain=pop_film - pop_base,
        pop_base=pop_base,
        pop_film=pop_film,
        plain=stat("imputation_accuracy", "ablation", "no_bias")["mean"],
        small_model=small_model,
        small_plain=small_plain,
        small_n=small_n,
        ablation_n=ablation_n,
        ablation_model=ablation_model,
        ablation_plain=ablation_plain,
        ablation_seeds=ablation_seeds,
        scaling_seeds=len(scaling_seeds) if not isinstance(scaling_seeds, int) else scaling_seeds,
        p_value=paired.p_value if paired else float("nan"),
        crossover_n=_crossover(scaling, eval_rate),
        attn_plain=stat("attention_vs_r2_pearson", "ablation", "no_bias")["mean"],
        block_missing=bool(base.get("block_missing", False)),
        block_len=int(base.get("block_len", 8)),
        strong=strong_entry.get("accuracy", float("nan")),
        strong_std=strong_entry.get("accuracy_std", 0.0),
        strong_delta=strong_paired.mean_delta if strong_paired else float("nan"),
        strong_p=strong_paired.p_value if strong_paired else float("nan"),
        strong_top_k=strong_entry.get("top_k", 0),
    )


# --------------------------------------------------------------------------- #
# Copy
# --------------------------------------------------------------------------- #
def background_paragraphs(h: Headline) -> list[str]:
    return [
        "Nearby DNA variants are often inherited together. That non-random pairing is called "
        "linkage disequilibrium (LD). Geneticists measure it with r²: a score from 0 (independent) "
        "to 1 (always travel together). Standard pipelines precompute every pairwise r² and store "
        "an L × L table, then rebuild that table for every new cohort, window and frequency cutoff.",
        "Imputation is guessing a genotype the lab did not observe. When neighbouring SNPs drop "
        "out together — as they do in genotyping-by-sequencing — a stored list of LD partners "
        "vanishes with the site it was meant to rescue. ldAttention is a small add-on that "
        "teaches attention that LD structure, so it can still look across the rest of the window, "
        "without ever building the r² table.",
    ]


def approach_bullets(h: Headline | None = None) -> list[tuple[str, str, str]]:
    return [
        ("What it does", "Before attention picks which variants to look at, we add two learned numbers — an LD bias — to those scores.", GREEN),
        ("Distance term", "Nearby sites get a boost. LD fades with genomic distance; this term learns that fade.", GREEN),
        ("Genotype term", "Sites whose alleles move together get a boost — a correlation-like pairwise score, learned from the data.", GREEN),
        ("Where it sits", "A drop-in for any standard attention layer (PyTorch SDPA or MultiheadAttention). No extra preprocessing stage.", BLUE),
        ("What it removes", "The stored L × L r² table. Nothing to build, store, or rebuild when the cohort or the window changes.", RED),
    ]


def methods_note(h: Headline) -> str:
    hide = (
        f"Hide {100 * h.mask_rate:.0f}% of genotypes in {h.block_len}-SNP blocks (as in GBS)."
        if h.block_missing
        else f"Hide {100 * h.mask_rate:.0f}% of genotypes."
    )
    return (
        f"{h.n_sites} SNPs × {h.n_individuals} people (msprime, MAF ≥ 5%). {hide} "
        f"Accuracy = % guessed right on held-out people. A point = 1 pp of that. "
        f"Same holes, {h.n_seeds} seeds."
    )


def conclusion_takeaways(h: Headline) -> list[tuple[str, str, str]]:
    """The three things a passer-by should get without stopping to read.

    Each one has to sit on a single line at ``SZ_BODY`` or the renderer shrinks
    the whole block below the size of the body copy, so keep them short; the
    supporting detail lives in the figure caption and the scope note.
    """
    return [
        (
            "Beats explicit LD",
            f"{h.pct(h.model)} of hidden genotypes right, "
            f"{100 * h.delta:+.1f} pts vs the usual r² pipeline.",
            GREEN,
        ),
        (
            "Recovers LD",
            f"attention matches the true r² table (r = {h.attn_r2:.2f}), never shown it.",
            BLUE,
        ),
        (
            "Nothing to precompute",
            "no stored L × L table, no rebuild when the cohort or window changes.",
            RED,
        ),
    ]


def conclusion_note(h: Headline) -> str:
    """The caveats, stated plainly rather than hidden inside the claim."""
    scope = (
        f"below ~{h.crossover_n} individuals explicit LD still wins, and "
        if h.crossover_n
        else ""
    )
    window = ""
    tuned = (
        f" Fully-tuned all-partner control: {h.pct(h.strong)} ({100 * h.strong_delta:+.1f} pts)."
        if h.has_strong else ""
    )
    return (
        f"Accuracy = % of hidden genotypes right on new people; a point = 1 pp of that. "
        f"{scope}The layer's edge: beat the r² pipeline without building that table (a) "
        f"and reconstruct the table in attention (e).{tuned}{window}"
    )


def figure_caption(h: Headline) -> str:
    panel_b = (
        f"explicit LD leads below ~{h.crossover_n} individuals, attention above"
        if h.crossover_n
        else "ahead of explicit LD at every cohort size"
    )
    gain = h.ablation_model - h.ablation_plain
    scaling_n = h.scaling_seeds or 3
    seed_note = f"{h.n_seeds} seeds (a, c, d)"
    extras = []
    if scaling_n != h.n_seeds:
        extras.append(f"{scaling_n} (b)")
    if extras:
        seed_note += "; " + "; ".join(extras)
    return (
        f"Figure 1. Accuracy = % hidden genotypes right on held-out people; a point = 1 pp. "
        f"a) vs usual and fully-tuned r² pipelines. b) {panel_b}. "
        "c) More genotypes hidden. d) By how common the variant is (MAF). "
        f"e) True r² vs attention. Mean ± s.d., {seed_note}."
    )
