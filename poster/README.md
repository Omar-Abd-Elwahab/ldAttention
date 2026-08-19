# ECCB 2026 Poster — ldAttention

## Files

| File | Use |
|---|---|
| `ECCB2026_ldAttention_poster.pdf` | Send to the print shop |
| `ECCB2026_ldAttention_poster.pptx` | Edit in PowerPoint |
| `SCRIPT_3MIN.md` | Three-minute talk, plus a 30-second version and expected questions |
| `IMPROVEMENTS.md` | What was tried to raise accuracy, and what to do next |

Canvas is the IID template's 46 × 36 in.

## How the build fits together

```
results/                         ← numbers, written by run_experiments.py
   │
   ├── scripts/poster_layout.py  ← single source of truth: geometry + copy
   │        │
   │        ├── scripts/make_poster_figures.py → poster/fig_*.png
   │        ├── scripts/build_poster.py        → .pptx  (editable)
   │        ├── scripts/render_poster_pdf.py   → .pdf   (print)
   │        └── scripts/make_script.py         → SCRIPT_3MIN.md
```

Nothing is hardcoded: every headline number in the copy, the figures and the talk
is read from `results/` at build time, so they cannot drift apart. Change the
wording or the grid in `poster_layout.py` and both outputs follow.

The PDF is composed directly rather than converted, because there is no
LibreOffice on this machine. Every text block is measured against the width of
its box and the font size steps down until it fits, so overflow is impossible by
construction — which is what produced "Backgrou nd" and "Metho" in the previous
build.

## Layout

```
┌──────────────────────────────┬──────────────────────────────┐
│ TITLE                        │ ③ Methods  — protocol strip  │
│ ① Background                 │ ④ Results  — A/B main        │
│ ② Approach — bias schematic  │            — C–F strip       │
│            + design bullets  │            + caption         │
│                              │ ⑤ Conclusion                 │
├──────────────────────────────┴──────────────────────────────┤
│ Footer: author · affiliations · ECCB 2026 · logos · QR      │
└─────────────────────────────────────────────────────────────┘
```

## Figures

| Slot | Section | File | Content |
|---|---|---|---|
| Picture 10 | Approach | `fig_architecture.png` | LDAttentionBias schematic |
| Picture 11 | Methods | `fig_validation_pipeline.png` | Benchmark protocol |
| Picture 8 | Results | `fig_results_main.png` | A head-to-head · B cohort scaling |
| Picture 24 | Results | `fig_results_strip.png` | C ablation · D missingness · E MAF · F attention vs true LD |

Each figure is drawn at exactly the aspect ratio of the box it lands in, so it
fills the slot instead of letterboxing.

## Regenerate everything

```bash
cd /media/omar/Projects/ldAttention

# 1. the reported sweep — 128 SNPs × 1000 people, GBS-like block missingness
.venv312/bin/python3.12 scripts/run_experiments.py --device cuda --use_msprime \
    --block_missing --n_sites 128 --n_haplotypes 2000 --n_seeds 6 \
    --skip_population --skip_budget

# 2. the small-cohort 4-arm ablation behind panel C (100 individuals — this is
#    the size at which the bias is still load-bearing; 200+ saturates)
.venv312/bin/python3.12 scripts/run_experiments.py --out_dir results_small --device cuda \
    --n_haplotypes 200 --n_seeds 8 --scaling_seeds 0 --budget_seeds 0 \
    --mask_rate_sweep "" --use_msprime

# 3. the saturated explicit-LD control (no model retraining; same splits/masks)
#    MUST use the same device as the sweep -- eval masks come from a device generator
.venv312/bin/python3.12 scripts/strong_baseline_pass.py --device cuda

# 4. figures, then preflight, then poster and talk
cd scripts
../.venv312/bin/python3.12 make_poster_figures.py
../.venv312/bin/python3.12 check_poster.py     # gates the rest; exits non-zero on trouble
../.venv312/bin/python3.12 build_poster.py
../.venv312/bin/python3.12 render_poster_pdf.py
../.venv312/bin/python3.12 make_script.py
```

`check_poster.py` is worth running every time. It verifies that every headline
number is finite, that each copy function still exists and renders, that the
conclusion takeaways fit on one line at body size (otherwise the renderer
silently shrinks the whole section), and that every figure slot has its file. It
warns rather than fails when `strong_baseline.csv` is missing, since the poster
still builds — it just quotes the weaker control.

`make_poster_figures.py` also writes `fig_qr.png`, encoded from
`poster_layout.REPO_URL`, which both builders substitute for the template's QR.

To rehearse a build against a scratch sweep without touching the reported one,
set `LDATTENTION_RESULTS=/path/to/other/results`. The tight spots in the layout
(value labels near 1.0, the delta arrow, the crossover marker) only misbehave at
realistic accuracies, so a short smoke sweep will not exercise them —
`scripts/make_mock_results.py` writes a synthetic directory in the same schema
for that purpose:

```bash
.venv312/bin/python3.12 scripts/make_mock_results.py --out /tmp/mock
cd scripts && LDATTENTION_RESULTS=/tmp/mock ../.venv312/bin/python3.12 make_poster_figures.py
```

Never point a printed build at it.

## Before printing

- [ ] Rasterize and read the PDF at full size: `pdftoppm -r 100 -png ECCB2026_ldAttention_poster.pdf out`
- [ ] **Confirm the repository URL is live before it goes on a wall.** The QR is now generated
      from `poster_layout.REPO_URL` rather than inherited from the template, so it and the printed
      link always agree — but nothing here can check that the URL resolves
- [ ] Scan the printed QR with a phone, not just the PDF
- [ ] Add the ECCB poster board number once assigned
- [ ] Check the Université Laval / CRIV / Genome Canada / Génome Québec / IBIS logos are current
- [ ] The `.pptx` asks for Baskerville Old Face and Abadi; neither is installed here, so open it
      once on a machine that has them (or re-export the PDF from PowerPoint) if you want the
      exact template faces rather than the substituted serif/sans in the composed PDF

## Claims this poster does and does not make

Worth knowing before you stand next to it:

- **Does claim:** higher held-out accuracy than a *fully-tuned* explicit-LD control at the reported
  cohort size, on identical masked entries, with a paired Wilcoxon test behind it; recovery of true
  LD structure without ever seeing r²; better sample efficiency at small cohorts; no stored L × L
  artifact and no per-cohort preprocessing stage.
- **Does not claim** a margin over a weak control. Panel A shows two explicit-LD controls: the
  sparse pipeline as usually run (top-8 r² partners) and a saturated one given every partner and
  trained to convergence. The headline delta is quoted against the *stronger* of the two, because
  the sparse one keeps improving as you feed it more — see `results/baseline_strength.json` and
  §0.3 of `IMPROVEMENTS.md`. Reporting only the sparse control would have inflated the margin from
  ~1.5 points to ~4.1.
- **Does not claim** that the r² matrix is redundant in general. At 64 sites a control can regress
  on every other site, so sparse partner selection buys the baseline nothing here; that is only
  true because the window is short. §0.4 of `IMPROVEMENTS.md` explains why a longer window is the
  highest-value next experiment.
- **Does not claim:** that it beats explicit LD at every cohort size. It does not — below the
  crossover in panel B the explicit-LD pipeline is the better estimator, and the poster says so
  in the panel title, the caption, the scope note under the conclusion and the talk. This is the
  first thing a sharp visitor will probe, so lead with it rather than defend it.
- **Does not claim:** that it is faster than computing r². It is not — r² is a cheap one-off per
  cohort, while the bias is recomputed every batch. The saving is structural, not wall-clock.
- **Does not claim:** that the LD bias raises the accuracy ceiling. With a large cohort and a long
  schedule a plain transformer catches up; the bias buys sample efficiency, and panel C is drawn
  from the small-cohort sweep for exactly that reason.

### Where the honesty is enforced in code, not prose

Several claims are derived rather than typed, so they cannot survive a sweep that contradicts
them:

| Claim | Falls back to |
|---|---|
| Panel A title "Beats the explicit-LD pipeline" | "Head-to-head against both controls" if the delta is negative |
| Panel B title / crossover marker | Drawn only when the curves actually cross |
| Panel C title "Both terms contribute" | "Bias saturates at this size" if the gain is under 0.5 points |
| Conclusion + caption + talk crossover sentences | Omitted entirely when there is no crossover |

The paired Wilcoxon test is exact (`ldattention/stats.py`, enumerated over all 2ⁿ sign
assignments — no scipy dependency) and is computed from `results_raw.csv` at build time.
