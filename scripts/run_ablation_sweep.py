"""Ablation + multi-seed sweep for the LD-aware attention mechanism.

Runs the ground-truth validation (see ``validate_ld.py``) across several random
seeds and four ablation configurations:

- ``none``          : no LD bias (plain-transformer control arm)
- ``distance_only`` : only the log-distance decay term
- ``genotype_only`` : only the symmetric genotype-context term
- ``full``          : both terms (the full LD-aware mechanism)

For each (seed, config) it trains the imputation model, measures masked-imputation
accuracy, and correlates the learned attention *and* the isolated learned LD bias
against the ground-truth pairwise r^2 (with a pure genomic-distance prior as a
reference baseline).

Every artifact is persisted under ``results/`` so it can be dropped into the paper:

    results/
      figures/    per-(config,seed) true-r^2 vs learned-bias heatmaps
      arrays/     raw .npy matrices (r^2, learned attention, learned bias)
      per_run_metrics.csv     one row per (seed, config)
      summary.csv             mean +/- std per config
      summary.json            same, structured
      summary.md              paper-ready markdown table
      summary_correlation.png bar chart: LD recovery per config
      summary_accuracy.png    bar chart: imputation accuracy per config
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow importing the sibling validation module when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_ld as V  # noqa: E402

from ldattention.tasks.imputation import LDAwareImputationModel  # noqa: E402

CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("none", {"use_distance_bias": False, "use_genotype_bias": False}),
    ("distance_only", {"use_distance_bias": True, "use_genotype_bias": False}),
    ("genotype_only", {"use_distance_bias": False, "use_genotype_bias": True}),
    ("full", {"use_distance_bias": True, "use_genotype_bias": True}),
]

METRIC_KEYS = [
    "imputation_accuracy",
    "bias_vs_r2_pearson",
    "bias_vs_r2_spearman",
    "attention_vs_r2_pearson",
    "attention_vs_r2_spearman",
]


def run_one(
    config_flags: dict[str, bool],
    features: torch.Tensor,
    positions_t: torch.Tensor,
    labels: torch.Tensor,
    r2: np.ndarray,
    positions: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = LDAwareImputationModel(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_distance=1.0,
        **config_flags,
    ).to(device)

    acc = V.train(
        model, features, positions_t, labels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        mask_rate=args.mask_rate, generator=generator,
    )
    attn = V.extract_mean_attention(model, features, positions_t, args.batch_size)
    bias = V.extract_mean_bias(model, features, positions_t, args.batch_size)

    r2_flat = V.upper_offdiag(r2)
    metrics = {
        "imputation_accuracy": acc,
        "bias_vs_r2_pearson": V._pearson(V.upper_offdiag(bias), r2_flat),
        "bias_vs_r2_spearman": V._spearman(V.upper_offdiag(bias), r2_flat),
        "attention_vs_r2_pearson": V._pearson(V.upper_offdiag(attn), r2_flat),
        "attention_vs_r2_spearman": V._spearman(V.upper_offdiag(attn), r2_flat),
    }
    return metrics, attn, bias


def aggregate(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for cfg_name, _ in CONFIGS:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        summary[cfg_name] = {}
        for key in METRIC_KEYS:
            vals = np.array([r[key] for r in cfg_rows], dtype=np.float64)
            summary[cfg_name][key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary


def save_tables(rows: list[dict], summary: dict, dist_baseline: dict, out_dir: Path) -> None:
    # Per-run CSV.
    with open(out_dir / "per_run_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "config", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)

    # Aggregated CSV.
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", *[f"{k}_mean" for k in METRIC_KEYS], *[f"{k}_std" for k in METRIC_KEYS]])
        for cfg_name, _ in CONFIGS:
            s = summary[cfg_name]
            writer.writerow(
                [cfg_name]
                + [f"{s[k]['mean']:.4f}" for k in METRIC_KEYS]
                + [f"{s[k]['std']:.4f}" for k in METRIC_KEYS]
            )

    # JSON (includes distance baseline reference).
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"configs": summary, "distance_prior_baseline": dist_baseline}, f, indent=2)

    # Paper-ready markdown table.
    with open(out_dir / "summary.md", "w") as f:
        f.write("# LD-aware attention: ablation + multi-seed results\n\n")
        f.write("Mean +/- std across seeds. Correlations are against ground-truth pairwise r^2.\n\n")
        f.write("| config | imputation acc | attention vs r2 (Pearson) | learned bias vs r2 (Pearson) |\n")
        f.write("|---|---|---|---|\n")
        for cfg_name, _ in CONFIGS:
            s = summary[cfg_name]
            f.write(
                f"| {cfg_name} "
                f"| {s['imputation_accuracy']['mean']:.3f} +/- {s['imputation_accuracy']['std']:.3f} "
                f"| {s['attention_vs_r2_pearson']['mean']:+.3f} +/- {s['attention_vs_r2_pearson']['std']:.3f} "
                f"| {s['bias_vs_r2_pearson']['mean']:+.3f} +/- {s['bias_vs_r2_pearson']['std']:.3f} |\n"
            )
        f.write(
            f"\nReference: pure genomic-distance prior vs r^2 (Pearson) = "
            f"{dist_baseline['pearson_mean']:+.3f} +/- {dist_baseline['pearson_std']:.3f}, "
            f"(Spearman) = {dist_baseline['spearman_mean']:+.3f} +/- {dist_baseline['spearman_std']:.3f}.\n"
        )


def save_summary_plots(summary: dict, dist_baseline: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figure] matplotlib not installed; skipping summary plots")
        return

    names = [c for c, _ in CONFIGS]
    x = np.arange(len(names))

    # Correlation bar chart (attention + bias) with distance baseline line.
    attn_mean = [summary[c]["attention_vs_r2_pearson"]["mean"] for c in names]
    attn_std = [summary[c]["attention_vs_r2_pearson"]["std"] for c in names]
    bias_mean = [summary[c]["bias_vs_r2_pearson"]["mean"] for c in names]
    bias_std = [summary[c]["bias_vs_r2_pearson"]["std"] for c in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.38
    ax.bar(x - w / 2, attn_mean, w, yerr=attn_std, capsize=4, label="learned attention")
    ax.bar(x + w / 2, bias_mean, w, yerr=bias_std, capsize=4, label="learned LD bias")
    ax.axhline(
        dist_baseline["pearson_mean"], color="k", ls="--", lw=1,
        label=f"distance prior ({dist_baseline['pearson_mean']:+.2f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Pearson correlation with true r^2")
    ax.set_title("LD recovery by configuration (higher = more LD-aware)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_correlation.png", dpi=130)
    plt.close(fig)

    # Imputation accuracy bar chart.
    acc_mean = [summary[c]["imputation_accuracy"]["mean"] for c in names]
    acc_std = [summary[c]["imputation_accuracy"]["std"] for c in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, acc_mean, 0.6, yerr=acc_std, capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("masked-imputation accuracy")
    ax.set_title("Imputation accuracy by configuration")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_accuracy.png", dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.n_seeds)) if args.seeds is None else args.seeds
    print(f"[setup] device={device.type} seeds={seeds} configs={[c for c, _ in CONFIGS]}")

    rows: list[dict] = []
    dist_pearsons: list[float] = []
    dist_spearmans: list[float] = []

    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if args.use_msprime:
            haplotypes, positions = V.simulate_haplotypes_msprime(args.n_haplotypes, args.n_sites, seed)
        else:
            haplotypes, positions = V.simulate_haplotypes_copying(
                args.n_haplotypes, args.n_sites, args.n_founders, args.rho, args.mutation_rate, rng
            )
        r2 = V.compute_true_r2(haplotypes)
        np.save(out_dir / "arrays" / f"r2_seed{seed}.npy", r2)

        # Distance-prior reference (config-independent).
        pos_dist = np.abs(positions[:, None] - positions[None, :])
        r2_flat = V.upper_offdiag(r2)
        dist_flat = V.upper_offdiag(-pos_dist)
        dist_pearsons.append(V._pearson(dist_flat, r2_flat))
        dist_spearmans.append(V._spearman(dist_flat, r2_flat))

        features, positions_t, labels = V.build_dataset(haplotypes, positions, device)

        for cfg_name, flags in CONFIGS:
            print(f"  --- config: {cfg_name} ---")
            metrics, attn, bias = run_one(
                flags, features, positions_t, labels, r2, positions, args, device, seed
            )
            rows.append({"seed": seed, "config": cfg_name, **{k: round(v, 6) for k, v in metrics.items()}})
            np.save(out_dir / "arrays" / f"{cfg_name}_seed{seed}_attn.npy", attn)
            np.save(out_dir / "arrays" / f"{cfg_name}_seed{seed}_bias.npy", bias)
            V.save_figure(r2, bias, str(out_dir / "figures" / f"heatmap_{cfg_name}_seed{seed}.png"))
            print(
                f"    acc={metrics['imputation_accuracy']:.4f} "
                f"attn_r2(Pear)={metrics['attention_vs_r2_pearson']:+.4f} "
                f"bias_r2(Pear)={metrics['bias_vs_r2_pearson']:+.4f}"
            )

    summary = aggregate(rows)
    dist_baseline = {
        "pearson_mean": float(np.mean(dist_pearsons)),
        "pearson_std": float(np.std(dist_pearsons)),
        "spearman_mean": float(np.mean(dist_spearmans)),
        "spearman_std": float(np.std(dist_spearmans)),
    }
    save_tables(rows, summary, dist_baseline, out_dir)
    save_summary_plots(summary, dist_baseline, out_dir)

    print("\n==================== SWEEP SUMMARY (mean +/- std) ====================")
    for cfg_name, _ in CONFIGS:
        s = summary[cfg_name]
        print(
            f"  {cfg_name:14s} acc={s['imputation_accuracy']['mean']:.3f}+/-{s['imputation_accuracy']['std']:.3f} "
            f"attn_r2={s['attention_vs_r2_pearson']['mean']:+.3f} "
            f"bias_r2={s['bias_vs_r2_pearson']['mean']:+.3f}"
        )
    print(f"  distance-prior baseline vs r^2 (Pearson) = {dist_baseline['pearson_mean']:+.3f}")
    print(f"\n[done] all artifacts written to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation + multi-seed sweep with saved artifacts.")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seed list (overrides n_seeds)")
    parser.add_argument("--n_haplotypes", type=int, default=600)
    parser.add_argument("--n_sites", type=int, default=48)
    parser.add_argument("--n_founders", type=int, default=5)
    parser.add_argument("--rho", type=float, default=15.0)
    parser.add_argument("--mutation_rate", type=float, default=0.005)
    parser.add_argument("--use_msprime", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mask_rate", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    main()
