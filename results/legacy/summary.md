# LD-aware attention: ablation + multi-seed results

Mean +/- std across seeds. Correlations are against ground-truth pairwise r^2.

| config | imputation acc | attention vs r2 (Pearson) | learned bias vs r2 (Pearson) |
|---|---|---|---|
| none | 0.444 +/- 0.040 | -0.080 +/- 0.018 | +0.000 +/- 0.000 |
| distance_only | 0.651 +/- 0.011 | +0.185 +/- 0.080 | +0.155 +/- 0.072 |
| genotype_only | 0.445 +/- 0.012 | -0.028 +/- 0.023 | +0.068 +/- 0.032 |
| full | 0.592 +/- 0.072 | +0.191 +/- 0.128 | +0.196 +/- 0.054 |

Reference: pure genomic-distance prior vs r^2 (Pearson) = +0.265 +/- 0.016, (Spearman) = +0.329 +/- 0.015.
