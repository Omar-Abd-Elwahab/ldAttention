"""Locate the regime where the LD bias actually earns its keep.

An inductive bias buys the most where the data cannot simply reveal the
structure on its own: small cohorts, long windows, and heavy missingness. This
probe sweeps those three axes with the bias on and off so the reported ablation
is run in a regime the claim can honestly be made in -- rather than one where a
plain transformer already saturates the task.

Not part of the reported results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from ldattention.validation import RunConfig, run_config

BASE = dict(
    use_msprime=True,
    hidden_dim=128,
    num_heads=8,
    num_layers=4,
    dropout=0.1,
    genotype_rank=32,
    epochs=150,
    lr=1.5e-3,
    batch_size=64,
    baseline_epochs=150,
    eval_every=10,
    position_frequencies=16,
    mask_rate_sweep=(),
)

AXES: dict[str, list[dict]] = {
    "cohort": [
        dict(n_haplotypes=n, n_sites=64, mask_rate=0.3) for n in (200, 400, 800, 1600)
    ],
    "length": [
        dict(n_haplotypes=1400, n_sites=L, mask_rate=0.3) for L in (64, 128, 256)
    ],
    "missingness": [
        dict(n_haplotypes=1400, n_sites=64, mask_rate=m) for m in (0.3, 0.5, 0.7, 0.9)
    ],
    "hard": [
        dict(n_haplotypes=n, n_sites=L, mask_rate=m)
        for n, L, m in ((300, 128, 0.5), (300, 256, 0.5), (600, 256, 0.7), (300, 128, 0.7))
    ],
}

ARMS = {"no_bias": (False, False), "both": (True, True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", type=str, default="cohort", choices=sorted(AXES))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed_offset", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] axis={args.axis} device={device} seeds={args.seeds}", flush=True)

    rows = []
    seeds = [args.seed_offset + s for s in range(args.seeds)]
    for point, seed in itertools.product(AXES[args.axis], seeds):
        for arm, (dist, geno) in ARMS.items():
            cfg = RunConfig(
                name=arm, use_distance_bias=dist, use_genotype_bias=geno, **{**BASE, **point}
            )
            start = time.perf_counter()
            res = run_config(cfg, seed=seed, device=device)
            row = {
                "arm": arm,
                "seed": seed,
                "seconds": time.perf_counter() - start,
                **point,
                **{k: v for k, v in res.items() if not k.startswith("_")},
            }
            rows.append(row)
            tag = " ".join(f"{k}={v}" for k, v in point.items())
            print(
                f"  {tag:<48s} {arm:9s} seed={seed} ({row['seconds']:5.1f}s) "
                f"test={row['imputation_accuracy']:.4f} "
                f"explicitLD={row['baseline_explicit_ld_accuracy']:.4f} "
                f"majority={row['baseline_majority_accuracy']:.4f} "
                f"attn~r2={row['attention_vs_r2_pearson']:+.3f}",
                flush=True,
            )

    out = Path(args.out or f"results/probe_{args.axis}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'point':<48s} {'no_bias':>9s} {'both':>9s} {'gain':>8s} {'explicitLD':>11s} {'majority':>9s}")
    for point in AXES[args.axis]:
        sel = [r for r in rows if all(r[k] == v for k, v in point.items())]
        if not sel:
            continue
        avg = lambda arm, key: sum(  # noqa: E731
            r[key] for r in sel if r["arm"] == arm
        ) / max(sum(r["arm"] == arm for r in sel), 1)
        nb, bo = avg("no_bias", "imputation_accuracy"), avg("both", "imputation_accuracy")
        tag = " ".join(f"{k}={v}" for k, v in point.items())
        print(
            f"{tag:<48s} {nb:9.4f} {bo:9.4f} {bo - nb:+8.4f} "
            f"{avg('both', 'baseline_explicit_ld_accuracy'):11.4f} "
            f"{avg('both', 'baseline_majority_accuracy'):9.4f}"
        )
    print(f"\n[done] {out.resolve()}")


if __name__ == "__main__":
    main()
