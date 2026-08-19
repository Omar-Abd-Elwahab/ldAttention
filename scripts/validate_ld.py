"""Single-run ground-truth validation for LD-aware attention.

Thin CLI over :mod:`ldattention.validation`. Simulates haplotypes with known
recombination structure, computes true pairwise r^2, trains the imputation model,
and correlates the learned attention/bias against real LD. Saves a comparison
heatmap.

For the full ablation + multi-seed sweep used in the paper, see
``scripts/run_experiments.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ldattention.validation import RunConfig, run_config


def save_figure(r2: np.ndarray, attn: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figure] matplotlib not installed; skipping heatmap")
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
    fig.suptitle("LD-aware attention recovers true LD structure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type}")

    cfg = RunConfig(
        name="both",
        n_haplotypes=args.n_haplotypes,
        n_sites=args.n_sites,
        n_founders=args.n_founders,
        rho=args.rho,
        mutation_rate=args.mutation_rate,
        use_msprime=args.use_msprime,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_rate=args.mask_rate,
    )

    print("[run] simulate -> train -> correlate against ground-truth r^2 ...")
    res = run_config(cfg, seed=args.seed, device=device, verbose=True)

    print("\n==================== VALIDATION RESULTS ====================")
    print(f"  masked-imputation accuracy          : {res['imputation_accuracy']:.4f}")
    print(f"  learned LD bias    vs true r^2 (Pear): {res['bias_vs_r2_pearson']:+.4f}")
    print(f"  learned LD bias    vs true r^2 (Spr) : {res['bias_vs_r2_spearman']:+.4f}")
    print(f"  learned attention  vs true r^2 (Pear): {res['attention_vs_r2_pearson']:+.4f}")
    print(f"  learned attention  vs true r^2 (Spr) : {res['attention_vs_r2_spearman']:+.4f}")
    print(f"  distance baseline  vs true r^2 (Pear): {res['distance_baseline_vs_r2_pearson']:+.4f}")
    print("============================================================\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_figure(res["_r2_matrix"], res["_attention_matrix"], args.out)
        print(f"[figure] saved comparison heatmap to {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LD-awareness against ground-truth r^2.")
    parser.add_argument("--n_haplotypes", type=int, default=600)
    parser.add_argument("--n_sites", type=int, default=48)
    parser.add_argument("--n_founders", type=int, default=5)
    parser.add_argument("--rho", type=float, default=15.0, help="recombination scale (higher = faster LD decay)")
    parser.add_argument("--mutation_rate", type=float, default=0.005)
    parser.add_argument("--use_msprime", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mask_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/ld_validation.png")
    return parser.parse_args()


if __name__ == "__main__":
    main()
