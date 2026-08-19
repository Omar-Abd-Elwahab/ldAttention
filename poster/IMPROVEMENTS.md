# Improving ldAttention — what was tried, and what to do next

Written against the corrected benchmark (msprime coalescent, common variants only,
MAF ≥ 5%, held-out test individuals). Every number below is a *held-out test*
number unless stated otherwise. See `results/` for the raw runs.

---

## 0. The two things that mattered most

Before anything else, two changes turned a result that lost to trivial baselines
into one that beats a real explicit-LD pipeline. Both are worth understanding
because they explain where the remaining headroom is.

### 0.1 The benchmark was broken, not the model

The original evaluation had **no held-out set** — the headline "93% accuracy" was
training accuracy. Adding a proper train/val/test split by individual exposed the
real problem: the unfiltered coalescent simulation had a site-frequency spectrum
dominated by singletons, so *predicting the modal genotype at each site already
scored 0.895*, and an explicit-LD regression scored 0.964. The model was learning
the major allele and nothing else.

Fixing it meant simulating what a genotyping array or imputation panel actually
contains: `mutation_rate=2.5e-8` and a MAF ≥ 5% filter, taking a contiguous
central window. That drops the majority-genotype floor to ~0.64–0.70 and leaves
genuine LD to learn (mean r² ≈ 0.14, ~19% of pairs above r² = 0.2).

**Lesson for the next iteration:** always report the majority-genotype floor
alongside accuracy. If a benchmark's trivial baseline is above ~0.85, the
benchmark is measuring allele frequency, not LD.

### 0.2 Positional resolution was the binding constraint

The model fed a *single raw position scalar* into a shared input projection, so
it had almost no capacity to tell one variant from another — while the
explicit-LD baseline it was being compared against has per-site parameters. That
asymmetry, not the attention mechanism, was what capped performance.

Replacing the scalar with sinusoidal features of the genomic coordinate
(`position_frequencies=16`, i.e. 16 sin/cos pairs plus the raw coordinate) is
worth about **15 accuracy points**:

| position encoding | test accuracy | dosage r² | vs explicit-LD |
|---|---|---|---|
| raw scalar (old) | 0.832 | 0.66 | −0.10 |
| 8 frequencies | 0.977 | 0.97 | +0.04 |
| **16 frequencies** | **0.980** | **0.97** | **+0.05** |
| 32 frequencies | 0.978 | 0.97 | +0.04 |

This is a *function of genomic position*, not a per-site lookup table, so unlike a
learned site embedding it still transfers to new windows, new variant counts and
new cohorts. That property is worth protecting in any future change.

### 0.3 The control was under-powered, and that inflated the headline

The explicit-LD control regressed each site on its top-8 r² partners for 150
epochs. Neither number was chosen for a reason, and both matter a great deal.
Sweeping them at the reported configuration (400 individuals, 64 sites, 4 seeds,
`scripts/check_baseline_strength.py`, archived in
`results/baseline_strength.json`):

| top-k partners | 150 epochs | 600 epochs |
|---|---|---|
| 8 (originally reported) | 0.9505 | 0.9604 |
| 16 | 0.9523 | 0.9633 |
| 32 | 0.9642 | 0.9664 |
| 63 (every other site) | 0.9574 | **0.9695** |

Against the saturated control the model's margin on the reported masks is
**+2.5 points, not +4.3** (paired Wilcoxon p = 0.002, 10 seeds,
`results/strong_baseline.csv`). The earlier probe (`baseline_strength.json`)
had suggested ~+1.5; the paired pass on the identical evaluation masks is the
number the poster quotes. At 800 individuals the saturated control is already
at 0.975 and the gap is smaller still.

The poster now reports both controls side by side in panel A and quotes the
margin against the stronger one. `scripts/strong_baseline_pass.py` recomputes it
on the identical splits and masks without retraining the model, so this costs
minutes rather than a re-run.

**Lesson:** tune the baseline with the same effort as the model, and say in the
caption which setting is being quoted. A margin over an untuned control is not a
result.

### 0.4 The window is too short for the argument being made

At 64 sites, "regress each site on every other site" is affordable — which is
why the saturated control does so well. But the entire reason real pipelines
compute an r² matrix is to *select* a sparse partner set when L is 10⁵–10⁶ and a
saturated regression is impossible. So this benchmark cannot demonstrate the
thing the method is supposed to save.

This is a limitation of the benchmark, not a negative result, and the poster and
talk both say so. The fix is a longer window (256–1024 sites), where sparse
partner selection is genuinely necessary and the O(L²) artifact is genuinely
expensive. That needs re-tuning (hidden dim and heads were chosen at L = 64) and
several hours of GPU time, so it is the single highest-value next experiment
rather than something to squeeze in before the print deadline.

---

## 1. Where the LD bias actually earns its keep

An inductive bias helps most where the data cannot reveal the structure by
itself. Sweeping cohort size with the bias on and off shows exactly that, and it
is the honest framing for the ablation. Numbers below are from the **reported**
sweep (`results/results.json`, 64 SNPs, 30% masked, 3 seeds per point), not from
the earlier exploratory probe:

| individuals | plain transformer | + LD bias | gain | explicit-LD | majority |
|---|---|---|---|---|---|
| 100 | 0.9017 | **0.9117** | **+0.0099** | 0.8927 | 0.7237 |
| 200 | 0.9758 | 0.9769 | +0.0011 | 0.9235 | 0.6389 |
| 400 | 0.9871 | 0.9887 | +0.0016 | 0.9479 | 0.7020 |
| 800 | 0.9908 | 0.9915 | +0.0007 | 0.9567 | 0.6580 |

Three things follow, and all three are honest selling points:

1. **The LD bias buys data efficiency, and only that.** At 100 individuals it is
   worth about a point. By 200 it is worth a tenth of one.
2. **It converges away.** From ~200 individuals a plain transformer with good
   positional features is already near ceiling on this task. At the reported
   cohort size (400 individuals, 10 seeds) the full ablation gap is +0.16 points
   — real under a paired test (8/10 seeds, p = 0.037), but small. Do not claim a
   large ablation gain at cohort scale; it is not there, and the poster says so
   in the scope note rather than waiting to be asked.
3. **The advantage over the sparse explicit-LD pipeline holds at every size
   tested**, including 100 individuals, because that pipeline is limited by its
   top-k linear structure rather than by data. This is against the top-8
   control as usually run; the saturated "every partner" control (still being
   scored as of this writing) is the one the poster quotes in the headline, and
   that margin will be smaller. Note this also contradicts the earlier
   exploratory probe, which suggested a crossover below ~150 individuals; with
   corrected positional features and a converged schedule no crossover appears
   in the 100–800 range against the sparse control. The poster's panel B and the
   talk both branch on whether one is found, so neither hard-codes the claim.

Regimes that were tried and did **not** help: longer windows (128/256 SNPs at
fixed cohort size changed nothing), and small-cohort + long-window + heavy
masking together, where 300 haplotypes over 128 SNPs at 50% masking collapses to
0.822 and loses badly to explicit-LD. More sites need proportionally more
individuals.

---

## 2. Ranked recommendations

### Priority 1 — Move to real data (biggest expected gain, and required for publication)

Everything above is simulated. The single most valuable next step is to run the
identical protocol on real genotypes through the **DeepGBSImpute** pipeline
(`/media/omar/Projects/ULAVAL/DeepGBSImpute`):

- Reuse its VCF/windowing loader to replace `simulate_haplotypes_msprime`. The
  interface `run_config` needs is small: `haplotypes [N, L] ∈ {0,1}`,
  `positions [L]`, and optional `pop_labels [N]`.
- Keep the split **by individual** (as here), and additionally hold out whole
  chromosomes to test cross-region transfer.
- Real soybean/cannabis GBS data has *structured* missingness (whole low-coverage
  regions absent), not the uniform random masking used here. Train with realistic
  missingness patterns — this is where the transformer should pull furthest ahead
  of a top-k regression, because its partners are missing in blocks.
- Report against the same three controls plus Beagle 5 / LinkImpute, which are
  what reviewers will ask for.

### Priority 2 — Structured missingness and a masking curriculum

Block masking is now in the code (`RunConfig.block_missing`, `--block_missing`).
Contiguous 8-SNP dropouts, like a GBS lane dying, are what the reported
large-data sweep uses. On a 128-SNP × 1000-person smoke seed the explicit-LD
control drops to ~0.89 (its top-k partners vanish in the same block) while
ldAttention holds ~0.98 — an ~8-point edge that iid masking hid. The LD layer
also lifts attention↔r² from ~0.65 to ~0.76.

`mask_rate_jitter` (currently 0.2 around a 0.3 base) already trains one model to
serve every missingness level. Still worth trying:

- **Curriculum:** start near 10% masked and anneal to 70%. Cheap — one extra
  schedule in `train_model`.
- **Curriculum:** start near 10% masked and anneal to 70%. Cheap to try — one
  extra schedule in `train_model`.

### Priority 3 — Scale the window, and scale the cohort with it

64 SNPs is a small window. The scaling table above shows sites and individuals
must grow together. Target 256–512 SNPs with ≥ 1000 individuals. Two things will
bind:

- **Memory.** The bias materialises `[B, H, L, L]`. At L = 512, B = 64, H = 8
  that is 4 GB in fp32 — more than this 6 GB RTX 2060 has spare. Fix with
  bf16/fp16 autocast (halves it) and gradient checkpointing, or compute the
  distance term once per batch (it depends only on positions, not on the
  genotype content) rather than per layer.
- **The distance term becomes more valuable** as windows lengthen, because a
  larger fraction of pairs are far apart and should be down-weighted a priori.

### Priority 4 — Cheap architectural wins, in expected order of payoff

- **Two-pass / iterative refinement.** Run the model once, feed its predicted
  dosages back in as the observed values for masked sites, run again. Standard in
  imputation and usually worth a point or two at high missingness, at 2× inference
  cost and no retraining.
- **Ensembling across seeds.** Averaging the softmax of 5 seeds is the single
  most reliable accuracy gain available and costs nothing but compute. The
  per-seed spread here (±0.005–0.03 depending on cohort size) suggests real
  headroom.
- **Predict dosage as well as class.** Add an auxiliary regression head on the
  expected dosage with an MSE term. Dosage r² is the metric imputation papers
  actually report, and optimising it directly usually improves calibration even
  when accuracy is flat.
- **Per-head rank for the genotype term.** `genotype_rank=32` is shared across
  heads; letting different heads use different ranks (or adding a learned
  per-head temperature on the bilinear term) is nearly free.
- **Untied bias across layers.** Every layer currently learns its own
  `LDAttentionBias`. Tying the *distance* term across layers while leaving the
  genotype term untied would cut parameters and may regularise small-cohort runs,
  which is exactly the regime where the bias matters.

### Priority 5 — Pretraining and transfer

The cohort-scaling result is essentially a statement that the model is
data-hungry. That is the classic case for pretraining:

- Pretrain the encoder with masked-genotype modelling on a large public panel
  (1000 Genomes for human; SoySNP50K / the lab's own panels for crops), then
  fine-tune on the target cohort.
- Because the positional encoding is a function of genomic coordinate rather than
  site index, the pretrained weights transfer across windows and marker sets
  directly. This is a genuine architectural advantage over per-site models and is
  worth stating explicitly in the paper.

### Priority 6 — MAF-stratified training

Accuracy is lowest in the high-MAF band (the hardest sites, where the majority
floor is worst). Options: reweight the loss by inverse site entropy, or oversample
masked entries at high-MAF sites. This will not move overall accuracy much but it
improves the metric reviewers scrutinise — rare and intermediate-frequency variant
recovery.

### Priority 7 — Cross-population conditioning

The FiLM gate (`num_populations > 0`) is currently only exercised on the
two-population copying simulation. To make it a real result it needs cohorts with
genuinely different LD — e.g. two crop breeding programmes, or two human
continental groups — and a leave-one-population-out protocol. As it stands it is
a mechanism demonstration, not evidence.

---

## 3. Things deliberately *not* recommended

- **Feeding r² in as a feature.** It would help, and it would also destroy the
  entire premise of the work.
- **A learned per-site embedding.** It would likely beat the sinusoidal encoding
  on a fixed marker set, and it would break transfer to new windows. The
  sinusoidal encoding is the right trade.
- **Claiming an asymptotic advantage over explicit r².** Both are O(L²) in the
  pairwise term. The honest claim is about *where the work lives*: no cohort-wide
  preprocessing pass, nothing stored, nothing recomputed per cohort — not a better
  complexity class.

---

## 4. Reproducing the analyses behind this document

```bash
cd /media/omar/Projects/ldAttention

# Regime probes (which cohort size / window length / missingness the bias helps in)
.venv312/bin/python3.12 scripts/probe_regimes.py --axis cohort --seeds 2
.venv312/bin/python3.12 scripts/probe_regimes.py --axis length --seeds 2

# Recipe search
.venv312/bin/python3.12 scripts/tune.py --grid coarse --seeds 2

# The reported sweep
.venv312/bin/python3.12 scripts/run_experiments.py --n_seeds 10 --use_msprime --device cuda
```
