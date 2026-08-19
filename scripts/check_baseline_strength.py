"""Is the explicit-LD control a fair opponent, or a strawman?

The central claim is measured against a per-site regression on each
site's top-``k`` LD partners. ``k`` was set to 8 without justification, which is
the first thing a reviewer should attack: a baseline can always be beaten by
starving it. This sweeps ``k`` and the baseline's training budget on the exact
data the reported sweep uses, so the choice can be defended with a number rather
than an assurance.

Nothing here trains the transformer, so it is cheap enough to run alongside the
main sweep.

    .venv312/bin/python3.12 scripts/check_baseline_strength.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.baselines import (  # noqa: E402
    LDRegressionBaseline,
    majority_baseline,
    select_ld_partners,
)
from ldattention.validation import (  # noqa: E402
    RunConfig,
    build_dataset,
    compute_true_r2,
    haplotypes_for,
    make_eval_masks,
    simulate,
    split_indices,
    subset,
)


def score_k(cfg: RunConfig, seed: int, device: torch.device, ks: list[int],
            epochs_list: list[int]) -> dict[tuple[int, int], float]:
    """Held-out accuracy of the explicit-LD control for each (k, epochs)."""
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
    r2_train = compute_true_r2(train_haps)

    rate = cfg.eval_mask_rate if cfg.eval_mask_rate is not None else cfg.mask_rate
    n_sites = data["features"].shape[1]
    test_masks = make_eval_masks(
        test_data["features"].shape[0], n_sites, rate, cfg.n_eval_repeats, seed + 7717, device
    )

    out: dict[tuple[int, int], float] = {}
    out[(0, 0)] = majority_baseline(train_data["labels"], test_data["labels"], test_masks)["accuracy"]
    for k in ks:
        partner_idx = select_ld_partners(r2_train, min(k, n_sites - 1))
        for epochs in epochs_list:
            reg = LDRegressionBaseline(partner_idx, device=device, epochs=epochs)
            reg.fit(train_data["features"], train_data["labels"], rate, generator,
                    batch_size=cfg.batch_size)
            out[(k, epochs)] = reg.score(
                test_data["features"], test_data["labels"], test_masks
            )["accuracy"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_haplotypes", type=int, default=2000)
    ap.add_argument("--n_sites", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--epochs", type=int, nargs="+", default=[120, 300])
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_strength.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = RunConfig(
        name="baseline_probe", n_haplotypes=args.n_haplotypes, n_sites=args.n_sites,
        use_msprime=True, mask_rate_sweep=(),
    )

    per_seed = [score_k(cfg, s, device, args.ks, args.epochs) for s in range(args.seeds)]
    keys = sorted(per_seed[0], key=lambda kv: (kv[0], kv[1]))

    print(f"\nexplicit-LD control, {args.seeds} seeds, {args.n_haplotypes // 2} individuals, "
          f"{args.n_sites} sites\n")
    print(f"{'top_k':>6} {'epochs':>7} {'accuracy':>18}")
    rows = {}
    for k, epochs in keys:
        vals = [p[(k, epochs)] for p in per_seed]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        rows[f"k{k}_e{epochs}"] = {"mean": mean, "std": std, "top_k": k, "epochs": epochs}
        label = "majority" if k == 0 else ""
        print(f"{k if k else '-':>6} {epochs if epochs else '-':>7} "
              f"{mean:>9.4f} ± {std:.4f}  {label}")

    tuned = {kk: v for kk, v in rows.items() if v["top_k"] > 0}
    best = max(tuned.values(), key=lambda v: v["mean"])
    default = rows.get("k8_e120")
    print(f"\nbest configuration: top_k={best['top_k']}, epochs={best['epochs']} "
          f"-> {best['mean']:.4f}")
    if default:
        gap = best["mean"] - default["mean"]
        print(f"reported setting  : top_k=8, epochs=120 -> {default['mean']:.4f} "
              f"({gap:+.4f} vs best)")
        verdict = ("the reported baseline is at or near its best -- the comparison is fair"
                   if gap < 0.005 else
                   "a stronger baseline exists; raise baseline_top_k/baseline_epochs before reporting")
        print(f"verdict: {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"seeds": args.seeds, "n_individuals": args.n_haplotypes // 2,
         "n_sites": args.n_sites, "results": rows}, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
