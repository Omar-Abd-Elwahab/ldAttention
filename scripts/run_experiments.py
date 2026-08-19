"""Paper-grade experiment suite for LD-aware attention.

Runs two experiments across multiple seeds and saves *all* figures and numeric
results to disk (default: ``results/``), ready to drop into the paper.

Every accuracy is measured on held-out test individuals, on masking draws that
are shared with two controls scored on the identical entries:

    majority       per-site modal genotype (the allele-frequency floor)
    explicit_ld    the O(L^2 N) pipeline this work replaces: build the full
                   pairwise r^2 matrix, keep each site's top-k LD partners, fit
                   a per-site multinomial logistic regression over them

Experiment A -- bias-term ablation (single population)
    no_bias        : plain transformer control (no LD bias)
    distance_only  : distance-decay term only
    genotype_only  : genotype-context term only
    both           : full LD-aware bias

Experiment B -- population structure
    both           : full bias, no population conditioning
    both_pop_film  : full bias + per-population FiLM conditioning
    (data simulated with 2 populations that have different LD structure)

Experiment C -- cohort scaling
    The same two arms (``no_bias`` / ``both``) across cohort sizes. An inductive
    bias should pay for itself where the data cannot reveal the structure on its
    own, so this is where the LD bias is expected to matter and where it is
    expected to stop mattering. Run with fewer seeds than A and B.

Saved artifacts
    results/results_raw.csv        per-run metrics (one row per config x seed)
    results/results_summary.csv    mean +/- std per config
    results/results.json           everything, machine-readable
    results/config.json            the exact settings used (including device)
    results/mask_rate_sweep.csv    accuracy vs missingness, model and controls
    results/fig_ablation_*.png     per-config true-r^2 vs learned-attention heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from ldattention.validation import RunConfig, run_config

METRIC_KEYS = [
    "imputation_accuracy",
    "dosage_r2",
    "train_accuracy",
    "val_accuracy",
    "baseline_majority_accuracy",
    "baseline_explicit_ld_accuracy",
    "model_minus_explicit_ld",
    "model_minus_majority",
    "explicit_r2_seconds",
    "accuracy_maf_low",
    "accuracy_maf_mid",
    "accuracy_maf_high",
    "baseline_explicit_ld_accuracy_maf_low",
    "baseline_explicit_ld_accuracy_maf_mid",
    "baseline_explicit_ld_accuracy_maf_high",
    "baseline_majority_accuracy_maf_low",
    "baseline_majority_accuracy_maf_mid",
    "baseline_majority_accuracy_maf_high",
    "attention_vs_r2_pearson",
    "attention_vs_r2_spearman",
    "bias_vs_r2_pearson",
    "bias_vs_r2_spearman",
    "attention_vs_r2_partial_pearson",
    "bias_vs_r2_partial_pearson",
    "distance_baseline_vs_r2_pearson",
]

_MATRIX_KEYS = ("_r2_matrix", "_attention_matrix", "_bias_matrix", "_abs_dist")

# Only the full model needs the accuracy-vs-missingness curve; refitting the
# explicit-LD control at every level for every ablation arm would triple runtime
# for a figure that only reports the full model.
SWEEP_CONFIGS = {"both", "both_pop_film"}


def ablation_configs(base: dict, sweep: tuple[float, ...]) -> list[RunConfig]:
    def arm(name: str, distance: bool, genotype: bool) -> RunConfig:
        return RunConfig(
            name=name,
            use_distance_bias=distance,
            use_genotype_bias=genotype,
            mask_rate_sweep=sweep if name in SWEEP_CONFIGS else (),
            **base,
        )

    return [
        arm("no_bias", False, False),
        arm("distance_only", True, False),
        arm("genotype_only", False, True),
        arm("both", True, True),
    ]


def scaling_configs(base: dict, sizes: list[int]) -> list[RunConfig]:
    """Experiment C: the bias on and off, across cohort sizes."""
    scale_base = {k: v for k, v in base.items() if k != "n_haplotypes"}
    return [
        RunConfig(
            name=f"{arm}_n{n // 2}", use_distance_bias=distance, use_genotype_bias=genotype,
            n_haplotypes=n, mask_rate_sweep=(), **scale_base,
        )
        for n in sizes
        for arm, (distance, genotype) in (("no_bias", (False, False)), ("both", (True, True)))
    ]


def budget_configs(base: dict, budgets: list[int], n_haplotypes: int) -> list[RunConfig]:
    """Experiment D: the bias on and off, across training budgets.

    Separates two things the ablation conflates: whether the LD bias raises the
    accuracy *ceiling*, and whether it gets there *sooner*. Trained to
    convergence a plain transformer matches the biased one on this benchmark, so
    the honest question is how much training each needs.
    """
    budget_base = {k: v for k, v in base.items() if k not in {"epochs", "n_haplotypes"}}
    return [
        RunConfig(
            name=f"{arm}_e{epochs}", use_distance_bias=distance, use_genotype_bias=genotype,
            epochs=epochs, n_haplotypes=n_haplotypes, mask_rate_sweep=(), **budget_base,
        )
        for epochs in budgets
        for arm, (distance, genotype) in (("no_bias", (False, False)), ("both", (True, True)))
    ]


def population_configs(base: dict, sweep: tuple[float, ...]) -> list[RunConfig]:
    # The population experiment needs distinct per-population LD, which the
    # msprime single-population coalescent does not provide, so it always uses
    # the multi-population copying model regardless of the global --use_msprime.
    pop_base = {**base, "n_populations": 2, "use_msprime": False}
    return [
        RunConfig(
            name="both", use_distance_bias=True, use_genotype_bias=True, pop_film=False,
            mask_rate_sweep=(), **pop_base,
        ),
        RunConfig(
            name="both_pop_film", use_distance_bias=True, use_genotype_bias=True, pop_film=True,
            mask_rate_sweep=(), **pop_base,
        ),
    ]


def save_heatmap(r2: np.ndarray, attn: np.ndarray, title: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(r2, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Ground-truth LD (true r^2)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    an = (attn - attn.min()) / (attn.max() - attn.min() + 1e-12)
    im1 = axes[1].imshow(an, cmap="viridis")
    axes[1].set_title("Learned LD-aware attention")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("variant")
        ax.set_ylabel("variant")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def summarize(rows: list[dict]) -> dict:
    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(r["config"], []).append(r)
    summary: dict[str, dict] = {}
    for cfg_name, runs in by_config.items():
        summary[cfg_name] = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs if r.get(key) is not None and not np.isnan(r.get(key, np.nan))]
            if not vals:
                continue
            summary[cfg_name][key] = {
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "sem": (pstdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        # The controls are scored on the same masked entries as the model within
        # each seed, so the per-seed differences are paired and a win count is
        # meaningful rather than decorative.
        for control in ("explicit_ld", "majority"):
            diffs = [r[f"model_minus_{control}"] for r in runs if f"model_minus_{control}" in r]
            if diffs:
                summary[cfg_name][f"wins_vs_{control}"] = {
                    "n_wins": sum(d > 0 for d in diffs),
                    "n": len(diffs),
                    "mean_delta": mean(diffs),
                    "std_delta": pstdev(diffs) if len(diffs) > 1 else 0.0,
                }
    return summary


def run_suite(name: str, configs, seeds, device, out_dir: Path, verbose: bool):
    rows: list[dict] = []
    sweep_rows: list[dict] = []
    print(f"\n===== Experiment: {name} =====", flush=True)
    for cfg in configs:
        for seed in seeds:
            started = time.time()
            res = run_config(cfg, seed=seed, device=device, verbose=verbose)
            # Save figures for the first seed of each config.
            if seed == seeds[0]:
                save_heatmap(
                    res["_r2_matrix"],
                    res["_attention_matrix"],
                    title=f"{name}: {cfg.name} (seed {seed})",
                    out_path=out_dir / f"fig_{name}_{cfg.name}.png",
                )
                np.save(out_dir / f"arr_{name}_{cfg.name}_r2.npy", res["_r2_matrix"])
                np.save(out_dir / f"arr_{name}_{cfg.name}_attn.npy", res["_attention_matrix"])
                np.save(out_dir / f"arr_{name}_{cfg.name}_bias.npy", res["_bias_matrix"])
            for k in _MATRIX_KEYS:
                res.pop(k, None)
            for point in res.pop("mask_rate_sweep", []):
                sweep_rows.append({"experiment": name, "config": cfg.name, "seed": seed, **point})
            print(
                f"  [{name}] {cfg.name:14s} seed={seed} "
                f"acc={res['imputation_accuracy']:.4f} "
                f"majority={res['baseline_majority_accuracy']:.4f} "
                f"explicitLD={res['baseline_explicit_ld_accuracy']:.4f} "
                f"delta={res['model_minus_explicit_ld']:+.4f} "
                f"attn~r2={res['attention_vs_r2_pearson']:+.3f} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
            rows.append(res)
    return rows, summarize(rows), sweep_rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in columns})


def write_summary_csv(path: Path, summary: dict, experiment: str) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for cfg_name, metrics in summary.items():
            for metric, stats in metrics.items():
                if "mean" not in stats:
                    continue
                writer.writerow(
                    [experiment, cfg_name, metric, stats["mean"], stats["std"], stats["sem"], stats["n"]]
                )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n_seeds))
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[setup] device={device.type} ({device_name}) seeds={seeds} out_dir={out_dir.resolve()}", flush=True)

    base = {
        "n_haplotypes": args.n_haplotypes,
        "n_sites": args.n_sites,
        "n_founders": args.n_founders,
        "rho": args.rho,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "genotype_rank": args.genotype_rank,
        "position_frequencies": args.position_frequencies,
        "mask_rate_jitter": args.mask_rate_jitter,
        "label_smoothing": args.label_smoothing,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "mask_rate": args.mask_rate,
        "use_msprime": args.use_msprime,
        "min_maf": args.min_maf,
        "baseline_top_k": args.baseline_top_k,
        "baseline_epochs": args.baseline_epochs,
        "eval_every": args.eval_every,
        "block_missing": args.block_missing,
        "block_len": args.block_len,
    }
    sweep = tuple(float(x) for x in args.mask_rate_sweep.split(",")) if args.mask_rate_sweep else ()

    started = time.time()
    ablation_rows, ablation_summary, ablation_sweep = run_suite(
        "ablation", ablation_configs(base, sweep), seeds, device, out_dir, args.verbose
    )
    if args.skip_population:
        population_rows, population_summary, population_sweep = [], {}, []
    else:
        population_rows, population_summary, population_sweep = run_suite(
            "population", population_configs(base, sweep), seeds, device, out_dir, args.verbose
        )
    scaling_sizes = [int(x) for x in args.scaling_sizes.split(",") if x]
    scaling_seeds = seeds[: args.scaling_seeds]
    scaling_rows, scaling_summary, _ = run_suite(
        "scaling", scaling_configs(base, scaling_sizes), scaling_seeds, device, out_dir, args.verbose
    )
    if args.skip_budget:
        budget_rows, budget_summary = [], {}
    else:
        budgets = [int(x) for x in args.budget_epochs.split(",") if x]
        budget_seeds = seeds[: args.budget_seeds]
        budget_rows, budget_summary, _ = run_suite(
            "budget",
            budget_configs(base, budgets, args.budget_haplotypes),
            budget_seeds, device, out_dir, args.verbose,
        )

    all_rows = (
        [{"experiment": "ablation", **r} for r in ablation_rows]
        + [{"experiment": "population", **r} for r in population_rows]
        + [{"experiment": "scaling", **r} for r in scaling_rows]
        + [{"experiment": "budget", **r} for r in budget_rows]
    )
    columns = ["experiment", "config", "seed", "best_epoch", *METRIC_KEYS]
    write_csv(out_dir / "results_raw.csv", all_rows, columns)

    summary_csv = out_dir / "results_summary.csv"
    with summary_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["experiment", "config", "metric", "mean", "std", "sem", "n"])
    write_summary_csv(summary_csv, ablation_summary, "ablation")
    write_summary_csv(summary_csv, population_summary, "population")
    write_summary_csv(summary_csv, scaling_summary, "scaling")
    write_summary_csv(summary_csv, budget_summary, "budget")

    sweep_rows = ablation_sweep + population_sweep
    if sweep_rows:
        write_csv(
            out_dir / "mask_rate_sweep.csv",
            sweep_rows,
            ["experiment", "config", "seed", "mask_rate", "model", "model_dosage_r2", "majority", "explicit_ld"],
        )

    meta = {
        "seeds": seeds,
        "device": device.type,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "base": base,
        "mask_rate_sweep": list(sweep),
        "scaling_sizes": scaling_sizes,
        "scaling_seeds": scaling_seeds,
        "cuda_version": torch.version.cuda,
        "total_seconds": time.time() - started,
    }
    payload = {
        "config": meta,
        "ablation": {"raw": ablation_rows, "summary": ablation_summary},
        "population": {"raw": population_rows, "summary": population_summary},
        "scaling": {"raw": scaling_rows, "summary": scaling_summary},
        "budget": {"raw": budget_rows, "summary": budget_summary},
        "mask_rate_sweep": sweep_rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    print("\n===================== SUMMARY (mean +/- std) =====================")
    for exp_name, summ in (
        ("ablation", ablation_summary),
        ("population", population_summary),
        ("scaling", scaling_summary),
        ("budget", budget_summary),
    ):
        print(f"\n[{exp_name}]")
        for cfg_name, metrics in summ.items():
            acc = metrics["imputation_accuracy"]
            maj = metrics["baseline_majority_accuracy"]
            eld = metrics["baseline_explicit_ld_accuracy"]
            wins = metrics.get("wins_vs_explicit_ld", {})
            print(
                f"  {cfg_name:16s} acc={acc['mean']:.4f}+/-{acc['std']:.4f}  "
                f"majority={maj['mean']:.4f}  explicitLD={eld['mean']:.4f}  "
                f"delta={wins.get('mean_delta', float('nan')):+.4f}  "
                f"wins={wins.get('n_wins', 0)}/{wins.get('n', 0)}"
            )
    print(f"\n[done] {time.time() - started:.0f}s; results written to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LD-aware attention ablation + multi-seed sweep.")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--n_seeds", type=int, default=10)
    # 800 haplotypes = 400 individuals: the scale of a real GBS panel, and the
    # point on the cohort-scaling curve where the model's advantage over the
    # explicit-LD pipeline is established rather than marginal.
    parser.add_argument("--n_haplotypes", type=int, default=2000)
    parser.add_argument("--n_sites", type=int, default=128)
    parser.add_argument("--n_founders", type=int, default=5)
    parser.add_argument("--rho", type=float, default=15.0)
    parser.add_argument("--min_maf", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    # The model underfits this task rather than overfitting it (validation
    # accuracy tracks training accuracy right up to the last epoch), so dropout
    # costs accuracy and a long schedule is worth more than regularisation.
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--genotype_rank", type=int, default=32)
    # A single raw position scalar leaves the shared input projection unable to
    # tell variants apart; expanding it over sin/cos frequencies is worth ~15
    # accuracy points and is what makes the model competitive at all.
    parser.add_argument("--position_frequencies", type=int, default=16)
    parser.add_argument("--mask_rate_jitter", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--scaling_sizes", type=str, default="400,800,2000")
    parser.add_argument("--scaling_seeds", type=int, default=3)
    parser.add_argument("--budget_epochs", type=str, default="50,100,200,400")
    parser.add_argument("--budget_seeds", type=int, default=5)
    parser.add_argument("--budget_haplotypes", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--baseline_top_k", type=int, default=8)
    parser.add_argument("--baseline_epochs", type=int, default=150)
    parser.add_argument("--mask_rate_sweep", type=str, default="0.1,0.2,0.3,0.5,0.7")
    parser.add_argument("--use_msprime", action="store_true")
    parser.add_argument("--block_missing", action="store_true",
                        help="Hide contiguous SNP blocks (GBS-like) instead of iid sites.")
    parser.add_argument("--block_len", type=int, default=8)
    parser.add_argument("--skip_population", action="store_true")
    parser.add_argument("--skip_budget", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
