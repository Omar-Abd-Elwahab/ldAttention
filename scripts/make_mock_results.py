"""Synthesise a results directory with realistic numbers, for layout rehearsal.

The poster's tight spots (value labels near 1.0, the delta arrow, the crossover
marker in panel B) only misbehave at the accuracies the real sweep produces, so
rehearsing the build against a short smoke run does not exercise them. This
writes a directory in the same schema with plausible values, purely so the
figures and the PDF can be checked before the real sweep lands.

Not part of the reported pipeline. Never point ``LDATTENTION_RESULTS`` at this
for anything that gets printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Targets taken from the ablation arm of the live sweep, which had finished
# no_bias and distance_only when this was written.
TARGETS = {
    "imputation_accuracy": (0.9832, 0.0043),
    "dosage_r2": (0.9610, 0.0090),
    "train_accuracy": (0.9910, 0.0030),
    "val_accuracy": (0.9820, 0.0040),
    "baseline_majority_accuracy": (0.6832, 0.0490),
    "baseline_explicit_ld_accuracy": (0.9419, 0.0100),
    "model_minus_explicit_ld": (0.0413, 0.0110),
    "model_minus_majority": (0.3000, 0.0490),
    "explicit_r2_seconds": (0.0120, 0.0020),
    "accuracy_maf_low": (0.9880, 0.0040),
    "accuracy_maf_mid": (0.9800, 0.0050),
    "accuracy_maf_high": (0.9770, 0.0060),
    "baseline_explicit_ld_accuracy_maf_low": (0.9650, 0.0090),
    "baseline_explicit_ld_accuracy_maf_mid": (0.9380, 0.0110),
    "baseline_explicit_ld_accuracy_maf_high": (0.9220, 0.0120),
    "baseline_majority_accuracy_maf_low": (0.8400, 0.0300),
    "baseline_majority_accuracy_maf_mid": (0.6600, 0.0400),
    "baseline_majority_accuracy_maf_high": (0.5600, 0.0400),
    "attention_vs_r2_pearson": (0.5450, 0.1100),
    "attention_vs_r2_spearman": (0.5100, 0.1100),
    "bias_vs_r2_pearson": (0.4800, 0.1200),
    "bias_vs_r2_spearman": (0.4500, 0.1200),
    "attention_vs_r2_partial_pearson": (0.3900, 0.1000),
    "bias_vs_r2_partial_pearson": (0.3400, 0.1000),
    "distance_baseline_vs_r2_pearson": (0.3100, 0.0800),
}

# The bias is worth a little at the main scale and more as the cohort shrinks.
ARM_OFFSET = {"no_bias": -0.0008, "distance_only": 0.0004, "genotype_only": 0.0006, "both": 0.0}

# (individuals): (model_with_bias, model_plain, explicit_ld). The explicit-LD
# pipeline leads on small cohorts and is overtaken -- the crossover panel B has
# to be able to draw.
SCALING = {
    100: (0.7450, 0.7150, 0.8730),
    200: (0.8300, 0.8040, 0.8950),
    400: (0.9080, 0.8950, 0.9160),
    800: (0.9620, 0.9600, 0.9330),
    1000: (0.9832, 0.9840, 0.9419),
}


def stat(mean: float, std: float, n: int) -> dict:
    return {"mean": mean, "std": std, "sem": std / (n ** 0.5), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mock")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(0)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Reuse the smoke run's matrices so panel F has something to draw.
    smoke = Path("/tmp/smoke")
    if smoke.exists():
        for npy in smoke.glob("arr_*.npy"):
            shutil.copy(npy, out / npy.name)

    n = args.seeds
    seeds = list(range(n))
    arms = ["no_bias", "distance_only", "genotype_only", "both"]

    raw_rows: list[dict] = []
    summary_rows: list[list] = []
    results: dict = {"config": {}, "ablation": {"raw": [], "summary": {}}}

    def add(experiment: str, config: str, metrics: dict[str, tuple[float, float]]) -> dict:
        summary: dict = {}
        for metric, (mean, std) in metrics.items():
            summary[metric] = stat(mean, std, n)
            summary_rows.append([experiment, config, metric, mean, std, std / (n ** 0.5), n])
        for seed in seeds:
            row = {"experiment": experiment, "config": config, "seed": seed, "best_epoch": 300}
            for metric, (mean, std) in metrics.items():
                row[metric] = round(rng.gauss(mean, std), 6)
            raw_rows.append(row)
        return summary

    for arm in arms:
        metrics = {
            k: (v[0] + (ARM_OFFSET[arm] if k in ("imputation_accuracy", "val_accuracy") else 0.0), v[1])
            for k, v in TARGETS.items()
        }
        # Keep the paired delta consistent with the accuracy it is derived from.
        metrics["model_minus_explicit_ld"] = (
            metrics["imputation_accuracy"][0] - metrics["baseline_explicit_ld_accuracy"][0], 0.011
        )
        summary = add("ablation", arm, metrics)
        deltas = [r["model_minus_explicit_ld"] for r in raw_rows
                  if r["config"] == arm and r["experiment"] == "ablation"]
        summary["wins_vs_explicit_ld"] = {
            "n_wins": sum(d > 0 for d in deltas), "n": n,
            "mean_delta": sum(deltas) / n, "std_delta": 0.011,
        }
        summary["wins_vs_majority"] = {"n_wins": n, "n": n, "mean_delta": 0.30, "std_delta": 0.05}
        results["ablation"]["summary"][arm] = summary

    scaling_summary = {}
    for size, (both, plain, explicit) in SCALING.items():
        for arm, acc in (("both", both), ("no_bias", plain)):
            name = f"{arm}_n{size}"
            scaling_summary[name] = add("scaling", name, {
                "imputation_accuracy": (acc, 0.012),
                "baseline_explicit_ld_accuracy": (explicit, 0.010),
                "baseline_majority_accuracy": (0.6832, 0.049),
                "model_minus_explicit_ld": (acc - explicit, 0.014),
            })
    results["scaling"] = {"summary": scaling_summary}

    pop_summary = {}
    for name, acc in (("both", 0.9410), ("both_pop_film", 0.9530)):
        pop_summary[name] = add("population", name, {"imputation_accuracy": (acc, 0.011)})
    results["population"] = {"summary": pop_summary}

    budget_summary = {}
    for epochs, (both, plain) in ((50, (0.905, 0.868)), (100, (0.951, 0.933)),
                                  (200, (0.974, 0.969)), (400, (0.983, 0.984))):
        for arm, acc in (("both", both), ("no_bias", plain)):
            name = f"{arm}_e{epochs}"
            budget_summary[name] = add("budget", name, {"imputation_accuracy": (acc, 0.010)})
    results["budget"] = {"summary": budget_summary}

    sweep_rows = []
    for rate, (model, explicit, majority) in (
        (0.1, (0.990, 0.960, 0.688)), (0.2, (0.987, 0.951, 0.688)),
        (0.3, (0.983, 0.942, 0.688)), (0.4, (0.976, 0.930, 0.688)),
        (0.5, (0.966, 0.914, 0.688)),
    ):
        for seed in seeds:
            sweep_rows.append({
                "experiment": "ablation", "config": "both", "seed": seed, "mask_rate": rate,
                "model": round(rng.gauss(model, 0.006), 6),
                "explicit_ld": round(rng.gauss(explicit, 0.010), 6),
                "majority": round(rng.gauss(majority, 0.049), 6),
            })
    with (out / "mask_rate_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)

    # Saturated explicit-LD control, as strong_baseline_pass.py would write it.
    # Deliberately close to the model, which is what the real sweep shows.
    strong_rows = []
    for seed in seeds:
        for rate, acc in ((0.1, 0.980), (0.2, 0.975), (0.3, 0.9695),
                          (0.4, 0.960), (0.5, 0.948)):
            strong_rows.append({
                "experiment": "ablation", "config": "both", "seed": seed,
                "mask_rate": rate, "top_k": 63, "epochs": 600,
                "accuracy": round(rng.gauss(acc, 0.007), 6),
                "accuracy_maf_low": round(rng.gauss(acc + 0.012, 0.007), 6),
                "accuracy_maf_mid": round(rng.gauss(acc - 0.004, 0.007), 6),
                "accuracy_maf_high": round(rng.gauss(acc - 0.010, 0.008), 6),
            })
    for size, (_, _, explicit) in SCALING.items():
        for seed in seeds[:4]:
            strong = min(explicit + 0.022, 0.985)
            strong_rows.append({
                "experiment": "scaling", "config": f"both_n{size}", "seed": seed,
                "mask_rate": 0.3, "top_k": 63, "epochs": 600,
                "accuracy": round(rng.gauss(strong, 0.008), 6),
                "accuracy_maf_low": "", "accuracy_maf_mid": "", "accuracy_maf_high": "",
            })
    with (out / "strong_baseline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strong_rows[0]))
        w.writeheader()
        w.writerows(strong_rows)

    base = {"n_sites": 64, "n_haplotypes": 2000, "mask_rate": 0.3, "epochs": 400}
    results["config"] = {"seeds": seeds, "device": "cuda", "device_name": "NVIDIA GeForce RTX 2060",
                         "base": base, "scaling_sizes": [2 * s for s in SCALING],
                         "scaling_seeds": 4, "total_seconds": 5400}
    results["mask_rate_sweep"] = sweep_rows

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(json.dumps({"base": base, "seeds": seeds}, indent=2))

    columns = ["experiment", "config", "seed", "best_epoch", *TARGETS]
    with (out / "results_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(raw_rows)

    with (out / "results_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "config", "metric", "mean", "std", "sem", "n"])
        w.writerows(summary_rows)

    print(f"mock results written to {out}")


if __name__ == "__main__":
    main()
