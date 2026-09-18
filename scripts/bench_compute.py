"""End-to-end time and memory cost of every method in the benchmark.

The accuracy tables say what each method buys. This says what each one costs, on
the same machine, at the same cohort size, across window lengths.

Four cost centres are separated because they are paid at different times:

``preprocess``  cohort-wide work before any fitting (the explicit pipeline's
                r^2 build and partner ranking). The learned bias has none.
``fit``         training / fitting to the budget used in the accuracy sweep.
``infer``       imputing the held-out cohort once, the cost paid per use.
``artifact``    bytes that must be stored and shipped with the fitted model.

Short phases are measured directly; long ones are measured per step or per epoch
and extrapolated to the sweep's budget, which is noted in the output. Timing is
independent of genotype values, so random haplotypes are used.

    .venv312/bin/python3.12 scripts/bench_compute.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldattention.baselines import (  # noqa: E402
    LDRegressionBaseline,
    majority_baseline,
    select_ld_partners,
)
from ldattention.tasks.imputation import LDAwareImputationModel  # noqa: E402
from ldattention.validation import (  # noqa: E402
    RunConfig,
    build_dataset,
    compute_true_r2,
    evaluate_model,
    make_eval_masks,
    sample_mask,
)

# Budgets actually used by the accuracy sweep, so cost and accuracy line up.
SWEEP_EPOCHS = {128: 250, 256: 200, 512: 150, 1024: 80}
SWEEP_BATCH = {128: 32, 256: 32, 512: 32, 1024: 16}
BASELINE_EPOCHS = 150


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    best = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync()
        best.append(time.perf_counter() - start)
    return float(np.median(best))


def bench_window(L: int, n_ind: int, device: torch.device) -> dict:
    epochs, batch = SWEEP_EPOCHS.get(L, 150), SWEEP_BATCH.get(L, 16)
    n_train = int(round(n_ind * 0.65))
    n_test = int(round(n_ind * 0.20))
    row: dict[str, float | int] = {"n_sites": L, "epochs": epochs, "batch_size": batch,
                                   "n_train": n_train, "n_test": n_test}

    rng = np.random.default_rng(0)
    hap = (rng.uniform(size=(2 * n_ind, L)) < 0.3).astype(np.int8)
    positions = np.sort(rng.uniform(size=L)).astype(np.float32)
    data = build_dataset(hap, positions, np.zeros(2 * n_ind, dtype=np.int64), device)
    train = {k: v[:n_train] for k, v in data.items()}
    test = {k: v[n_train : n_train + n_test] for k, v in data.items()}
    train_haps = hap[: 2 * n_train]

    cfg = RunConfig(
        name="bench", n_sites=L, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.0,
        genotype_rank=32, position_frequencies=16, batch_size=batch, mask_rate=0.3,
        block_missing=True, block_len=8, use_missing_channel=True, n_eval_repeats=5,
    )
    masks = make_eval_masks(n_test, L, 0.3, 5, 7717, device, True, 8)

    # ---- explicit-LD preprocessing: the cohort-wide pass before any fitting --
    row["r2_build_cpu_s"] = _time(lambda: compute_true_r2(train_haps), repeats=3)
    r2 = compute_true_r2(train_haps)
    row["r2_bytes"] = r2.nbytes
    row["partner_select_s"] = _time(lambda: select_ld_partners(r2, 8), repeats=3)

    # ---- explicit-LD fit + inference, at two partner counts ------------------
    for top_k in (8, min(64, L - 1)):
        tag = f"explicit_top{top_k}"
        partners = select_ld_partners(r2, top_k)
        gen = torch.Generator(device=device).manual_seed(0)
        reg = LDRegressionBaseline(partners, device=device, epochs=1)

        def one_epoch(reg=reg, gen=gen) -> None:
            reg.fit(train["features"], train["labels"], 0.3, gen,
                    batch_size=batch, block_missing=True, block_len=8)

        torch.cuda.reset_peak_memory_stats()
        per_epoch = _time(one_epoch, repeats=3)
        row[f"{tag}_fit_epoch_s"] = per_epoch
        row[f"{tag}_fit_total_s"] = per_epoch * BASELINE_EPOCHS
        row[f"{tag}_fit_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        row[f"{tag}_infer_s"] = _time(
            lambda reg=reg: reg.score(test["features"], test["labels"], masks), repeats=3
        )
        row[f"{tag}_artifact_bytes"] = int(
            reg.weight.numel() * 4 + reg.bias.numel() * 4 + partners.nbytes + r2.nbytes
        )
        del reg, partners
        torch.cuda.empty_cache()

    # ---- majority genotype ---------------------------------------------------
    row["majority_fit_infer_s"] = _time(
        lambda: majority_baseline(train["labels"], test["labels"], masks), repeats=3
    )
    row["majority_artifact_bytes"] = L * 8

    # ---- the two transformer arms -------------------------------------------
    for tag, kwargs in (
        ("plain", dict(use_distance_bias=False, use_genotype_bias=False)),
        ("ldattention", dict(use_distance_bias=True, use_genotype_bias=True)),
    ):
        model = LDAwareImputationModel(
            input_dim=3, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.0,
            genotype_rank=32, max_distance=1.0, position_frequencies=16, **kwargs
        ).to(device)
        row[f"{tag}_params"] = sum(p.numel() for p in model.parameters())
        row[f"{tag}_artifact_bytes"] = row[f"{tag}_params"] * 4

        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
        gen = torch.Generator(device=device).manual_seed(0)
        feats, labels = train["features"], train["labels"]
        pos = train["positions"]

        def one_step(model=model, opt=opt, gen=gen) -> None:
            idx = torch.randperm(n_train, generator=gen, device=device)[:batch]
            f2, p2, y = feats[idx], pos[idx], labels[idx]
            mask = sample_mask(f2.shape[0], L, 0.3, gen, device, True, 8)
            masked = f2.clone()
            masked[mask] = 0.0
            x = torch.cat([masked, mask.to(masked.dtype).unsqueeze(-1)], dim=-1)
            logits, _ = model(x, p2)
            loss = F.cross_entropy(logits[mask], y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.cuda.reset_peak_memory_stats()
        per_step = _time(one_step, repeats=10, warmup=3)
        steps_per_epoch = int(np.ceil(n_train / batch))
        row[f"{tag}_train_step_s"] = per_step
        row[f"{tag}_train_epoch_s"] = per_step * steps_per_epoch
        row[f"{tag}_train_total_s"] = per_step * steps_per_epoch * epochs
        row[f"{tag}_train_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30

        torch.cuda.reset_peak_memory_stats()
        row[f"{tag}_infer_s"] = _time(
            lambda model=model: evaluate_model(model, test, masks, cfg, use_population=False),
            repeats=3,
        )
        row[f"{tag}_infer_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30

        del model, opt
        torch.cuda.empty_cache()

    del data, train, test
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--n_individuals", type=int, default=1000)
    ap.add_argument("--out", type=str, default="results_scale/compute_cost.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[setup] device={device.type} ({name}) N={args.n_individuals}", flush=True)

    rows = []
    for L in args.lengths:
        row = bench_window(L, args.n_individuals, device)
        rows.append(row)
        print(f"  L={L:5d} done: ldattention train {row['ldattention_train_total_s'] / 60:6.1f} min, "
              f"plain {row['plain_train_total_s'] / 60:6.1f} min, "
              f"explicit top8 {row['explicit_top8_fit_total_s'] / 60:5.2f} min, "
              f"r2 build {row['r2_build_cpu_s']:.3f} s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": name, "n_individuals": args.n_individuals,
                               "sweep_epochs": SWEEP_EPOCHS, "baseline_epochs": BASELINE_EPOCHS,
                               "rows": rows}, indent=2))
    print(f"[done] {out.resolve()}")


if __name__ == "__main__":
    main()
