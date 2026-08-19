"""Sequential GPU hyperparameter probe used to pick the reported training recipe.

Not part of the reported results: this only searches for a recipe. Runs one job
at a time on the GPU (a single 6 GB card is the bottleneck, so fanning out with
a process pool would only cause contention).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ldattention.validation import RunConfig, run_config

BASE = dict(
    n_sites=64,
    use_msprime=True,
    batch_size=64,
    mask_rate=0.3,
    baseline_epochs=150,
    n_eval_repeats=3,
    eval_every=10,
    mask_rate_sweep=(),  # skip the missingness curve while tuning
)

GRID: dict[str, dict] = {
    "base_700": dict(n_haplotypes=700, hidden_dim=128, num_heads=8, num_layers=4, epochs=150, lr=2e-3, dropout=0.1),
    "700_long_nodrop": dict(n_haplotypes=700, hidden_dim=128, num_heads=8, num_layers=4, epochs=400, lr=2e-3, dropout=0.0),
    "2000_nodrop": dict(n_haplotypes=2000, hidden_dim=128, num_heads=8, num_layers=4, epochs=250, lr=2e-3, dropout=0.0),
    "4000_nodrop": dict(n_haplotypes=4000, hidden_dim=128, num_heads=8, num_layers=4, epochs=200, lr=2e-3, dropout=0.0),
    "4000_big": dict(n_haplotypes=4000, hidden_dim=192, num_heads=8, num_layers=4, epochs=200, lr=1.5e-3, dropout=0.0, genotype_rank=32),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", type=str, default="results/tuning.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device.type}")
    grid = {k: v for k, v in GRID.items() if not args.only or k in args.only.split(",")}

    rows = []
    for name, overrides in grid.items():
        for seed in range(args.seeds):
            cfg = RunConfig(name=name, **{**BASE, **overrides})
            t = time.time()
            res = run_config(cfg, seed=seed, device=device)
            res = {k: v for k, v in res.items() if not k.startswith("_")}
            res["seconds"] = time.time() - t
            rows.append(res)
            print(
                f"  {name:16s} seed={seed} acc={res['imputation_accuracy']:.4f} "
                f"train={res['train_accuracy']:.4f} val={res['val_accuracy']:.4f} "
                f"best_ep={res['best_epoch']:4d} majority={res['baseline_majority_accuracy']:.4f} "
                f"explicitLD={res['baseline_explicit_ld_accuracy']:.4f} "
                f"delta={res['model_minus_explicit_ld']:+.4f} ({res['seconds']:.0f}s)",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
