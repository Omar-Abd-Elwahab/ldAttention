"""Run the missing LDAttention (no_bias) mask-rate sweep for Figure 4A.

The primary ``results_large`` sweep only scored ``both`` (+Bias) at each
missingness rate (see ``SWEEP_CONFIGS`` in ``run_experiments.py``). This pass
retrains the no-bias arm under the same protocol and appends its sweep rows
to ``mask_rate_sweep.csv`` without rewriting other result files.

    .venv312/bin/python3.12 scripts/run_nobias_mask_sweep.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.validation import RunConfig, run_config  # noqa: E402

SWEEP_COLUMNS = [
    "experiment", "config", "seed", "mask_rate",
    "model", "model_dosage_r2", "majority", "explicit_ld",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_large")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--n_seeds", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    config = json.loads((out_dir / "config.json").read_text())
    base = dict(config["base"])
    sweep = tuple(float(x) for x in config.get("mask_rate_sweep", [0.1, 0.3, 0.5, 0.7]))
    seeds = list(range(args.n_seeds))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"[setup] device={device.type} seeds={seeds} sweep={list(sweep)} out={out_dir}",
        flush=True,
    )

    cfg = RunConfig(
        name="no_bias",
        use_distance_bias=False,
        use_genotype_bias=False,
        mask_rate_sweep=sweep,
        **base,
    )

    new_rows: list[dict] = []
    started = time.time()
    for seed in seeds:
        t0 = time.time()
        res = run_config(cfg, seed=seed, device=device, verbose=False)
        points = res.pop("mask_rate_sweep", [])
        for point in points:
            new_rows.append(
                {"experiment": "ablation", "config": "no_bias", "seed": seed, **point}
            )
        print(
            f"  no_bias seed={seed} acc={res['imputation_accuracy']:.4f} "
            f"sweep_points={len(points)} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    path = out_dir / "mask_rate_sweep.csv"
    existing = list(csv.DictReader(path.open())) if path.exists() else []
    # Drop any prior no_bias sweep rows so re-runs stay idempotent.
    kept = [r for r in existing if not (r["experiment"] == "ablation" and r["config"] == "no_bias")]
    merged = kept + [
        {c: row.get(c) for c in SWEEP_COLUMNS} for row in new_rows
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    print(
        f"[done] wrote {len(new_rows)} no_bias sweep rows "
        f"({len(kept)} prior rows kept) in {time.time() - started:.0f}s -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
