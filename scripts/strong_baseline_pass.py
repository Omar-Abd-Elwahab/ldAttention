"""Recompute the explicit-LD control at full strength, post hoc.

The reported sweep scores an explicit-LD control that regresses each site on its
top-8 r^2 partners for 150 epochs. Neither number was justified, and
``scripts/check_baseline_strength.py`` shows the control keeps improving with
more partners and a longer schedule -- so the reported margin over it is partly
an artifact of a weak opponent.

The control does not depend on the trained model, and the split and the
evaluation masks are fully determined by the seed, so the fair version can be
recomputed on the *identical* masked entries without retraining anything. That
is what this does. The poster then reports both: the sparse pipeline as it is
usually run, and this saturated version as the strongest linear control the
benchmark admits.

Must run on the same device as the sweep: ``make_eval_masks`` draws from a
device-specific torch generator, so a CPU pass would score different entries.

    .venv312/bin/python3.12 scripts/strong_baseline_pass.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.baselines import (  # noqa: E402
    LDRegressionBaseline,
    select_ld_partners,
)
from ldattention.validation import (  # noqa: E402
    MAF_BINS,
    RunConfig,
    build_dataset,
    compute_true_r2,
    haplotypes_for,
    make_eval_masks,
    minor_allele_freq,
    simulate,
    split_indices,
    subset,
)

FIELDS = [
    "experiment", "config", "seed", "mask_rate", "top_k", "epochs",
    "accuracy", "accuracy_maf_low", "accuracy_maf_mid", "accuracy_maf_high",
]


def strong_baseline(
    cfg: RunConfig, seed: int, device: torch.device, epochs: int, top_k: int | None,
    rates: list[float],
) -> list[dict]:
    """Score the saturated explicit-LD control on the run's own test masks.

    Mirrors ``run_config``'s data preparation exactly -- same simulate call, same
    rng consumption order, same split, same evaluation masks -- so the resulting
    accuracies are paired with the reported model accuracies seed by seed.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    sim = simulate(cfg, rng, seed)
    data = build_dataset(sim["haplotypes"], sim["positions"], sim["pop_labels"], device)
    n_individuals = data["features"].shape[0]
    train_idx, _, test_idx = split_indices(n_individuals, cfg.val_frac, cfg.test_frac, rng)
    train_data, test_data = subset(data, train_idx), subset(data, test_idx)

    haplotypes = sim["haplotypes"][: n_individuals * 2]
    train_haps = haplotypes_for(haplotypes, train_idx)
    maf = minor_allele_freq(train_haps)
    r2_train = compute_true_r2(train_haps)

    n_sites = data["features"].shape[1]
    k = (n_sites - 1) if not top_k else min(top_k, n_sites - 1)
    partner_idx = select_ld_partners(r2_train, k)

    rows = []
    for rate in rates:
        masks = make_eval_masks(
            test_data["features"].shape[0], n_sites, rate, cfg.n_eval_repeats,
            seed + 7717, device, cfg.block_missing, cfg.block_len,
        )
        reg = LDRegressionBaseline(partner_idx, device=device, epochs=epochs)
        reg.fit(train_data["features"], train_data["labels"], rate, generator,
                batch_size=cfg.batch_size,
                block_missing=cfg.block_missing, block_len=cfg.block_len)
        scored = reg.score(test_data["features"], test_data["labels"], masks)

        row = {"seed": seed, "mask_rate": rate, "top_k": k, "epochs": epochs,
               "accuracy": scored["accuracy"]}
        site_maf = maf[scored["sites"]]
        for label, low, high in MAF_BINS:
            sel = (site_maf >= low) & (site_maf < high)
            row[f"accuracy_maf_{label}"] = (
                float(scored["correct"][sel].mean()) if sel.any() else float("nan")
            )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--top_k", type=int, default=0, help="0 means every other site")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = Path(args.results)
    config = json.loads((results / "config.json").read_text())
    base = dict(config["base"])
    seeds = config["seeds"]
    device = torch.device(args.device)

    eval_rate = base.get("eval_mask_rate") or base["mask_rate"]
    sweep_rates = sorted({eval_rate, *config.get("mask_rate_sweep", [])})

    # The baseline is a property of the data, so one pass per seed covers every
    # ablation arm, and one per (cohort size, seed) covers the scaling curve.
    jobs: list[tuple[str, str, dict, int, list[float]]] = []
    for seed in seeds:
        jobs.append(("ablation", "both", dict(base), seed, sweep_rates))

    scaling_sizes = config.get("scaling_sizes", [])
    raw_scaling_seeds = config.get("scaling_seeds", 0)
    # run_experiments.py writes the actual seed list; older configs stored a count.
    scaling_seeds = (
        list(raw_scaling_seeds) if isinstance(raw_scaling_seeds, list)
        else seeds[: int(raw_scaling_seeds)]
    )
    for size in scaling_sizes:
        cfg_over = {**base, "n_haplotypes": size}
        for seed in scaling_seeds:
            jobs.append(("scaling", f"both_n{size // 2}", cfg_over, seed, [eval_rate]))

    print(f"[setup] device={device.type} epochs={args.epochs} "
          f"top_k={'all' if not args.top_k else args.top_k} jobs={len(jobs)}", flush=True)

    rows: list[dict] = []
    started = time.time()
    for i, (experiment, name, cfg_over, seed, rates) in enumerate(jobs, 1):
        cfg = RunConfig(name=name, mask_rate_sweep=(), **cfg_over)
        t0 = time.time()
        for row in strong_baseline(cfg, seed, device, args.epochs, args.top_k, rates):
            rows.append({"experiment": experiment, "config": name, **row})
        main_row = next(r for r in rows[-len(rates):] if r["mask_rate"] == rates[0])
        print(f"  [{i}/{len(jobs)}] {experiment:9s} {name:14s} seed={seed} "
              f"acc={main_row['accuracy']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    out = results / "strong_baseline.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})
    print(f"\n{len(rows)} rows -> {out}  ({time.time() - started:.0f}s total)")


if __name__ == "__main__":
    main()
