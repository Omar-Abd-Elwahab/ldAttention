"""Preprocessing-cost benchmark: explicit r² matrix vs the LDAttentionBias forward pass.

The claim under test is *not* that ldAttention avoids quadratic pairwise work --
any dense pairwise bias is O(L²). The claim is about where that work lives:

``explicit LD pipeline``
    A separate cohort-wide pass that must materialize and store an L x L r²
    matrix before training can start, and must be redone for every new cohort,
    window, or MAF threshold.

``ldAttention``
    No preprocessing pass at all. The pairwise term is produced inside the
    forward pass from tensors already resident on the accelerator, so its cost
    is an *increment* on a forward pass the pipeline was doing anyway.

Three quantities are timed across L = 64 ... 2048:

1. ``explicit_r2_cpu``   -- numpy cohort-wide r² build (what plink/numpy pipelines do).
2. ``explicit_r2_gpu``   -- the same build on GPU (a deliberately generous control).
3. ``ld_bias_forward``   -- LDAttentionBias forward for one epoch over the cohort.
4. ``model_forward``     -- full model forward with and without the bias, giving
   the LD-awareness overhead as a percentage of a forward pass.

Writes ``results/preprocessing_cost.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ldattention.models.ld_bias import LDAttentionBias
from ldattention.tasks.imputation import LDAwareImputationModel


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timeit(fn, device: torch.device, repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def explicit_r2_numpy(haplotypes: np.ndarray) -> np.ndarray:
    """Cohort-wide pairwise r², the way an explicit-LD preprocessing step builds it."""
    h = haplotypes.astype(np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    std = h.std(axis=0) + 1e-12
    h = h / std
    corr = (h.T @ h) / h.shape[0]
    return corr**2


def explicit_r2_torch(haplotypes: torch.Tensor) -> torch.Tensor:
    h = haplotypes - haplotypes.mean(dim=0, keepdim=True)
    h = h / (h.std(dim=0) + 1e-12)
    corr = (h.T @ h) / h.shape[0]
    return corr**2


def bench_length(
    seq_len: int,
    n_haplotypes: int,
    batch_size: int,
    hidden_dim: int,
    num_heads: int,
    device: torch.device,
    repeats: int,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    haplotypes = (rng.uniform(size=(n_haplotypes, seq_len)) < 0.3).astype(np.float32)
    n_individuals = n_haplotypes // 2
    row = {"seq_len": seq_len, "n_haplotypes": n_haplotypes, "n_individuals": n_individuals}

    row["explicit_r2_cpu_s"] = _timeit(lambda: explicit_r2_numpy(haplotypes), torch.device("cpu"), repeats)

    if device.type == "cuda":
        hap_t = torch.tensor(haplotypes, device=device)
        row["explicit_r2_gpu_s"] = _timeit(lambda: explicit_r2_torch(hap_t), device, repeats)
        del hap_t
        torch.cuda.empty_cache()

    bias = LDAttentionBias(hidden_dim=hidden_dim, num_heads=num_heads, max_distance=1.0).to(device)
    positions = torch.rand(batch_size, seq_len, device=device).sort(dim=1).values
    embeddings = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    @torch.no_grad()
    def one_bias_batch() -> None:
        bias(positions, embeddings)

    try:
        per_batch = _timeit(one_bias_batch, device, repeats)
        n_batches = max(int(np.ceil(n_individuals / batch_size)), 1)
        row["ld_bias_batch_s"] = per_batch
        row["ld_bias_epoch_s"] = per_batch * n_batches
    except torch.cuda.OutOfMemoryError:
        row["ld_bias_batch_s"] = float("nan")
        row["ld_bias_epoch_s"] = float("nan")
    del bias, embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Full-model forward with and without the bias -> LD-awareness overhead.
    feats = torch.randn(batch_size, seq_len, 3, device=device)
    pos3 = positions.unsqueeze(-1)
    for tag, kwargs in (
        ("model_plain_s", dict(use_distance_bias=False, use_genotype_bias=False)),
        ("model_ldaware_s", dict(use_distance_bias=True, use_genotype_bias=True)),
    ):
        model = LDAwareImputationModel(
            input_dim=3, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=1, dropout=0.0, **kwargs
        ).to(device)

        @torch.no_grad()
        def one_forward(m=model) -> None:
            m(feats, pos3)

        try:
            row[tag] = _timeit(one_forward, device, repeats)
        except torch.cuda.OutOfMemoryError:
            row[tag] = float("nan")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if row.get("model_plain_s"):
        row["ld_overhead_pct"] = 100.0 * (row["model_ldaware_s"] / row["model_plain_s"] - 1.0)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="results/preprocessing_cost.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--n_haplotypes", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} lengths={args.lengths}")
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)}")

    rows = [
        bench_length(
            L, args.n_haplotypes, args.batch_size, args.hidden_dim, args.num_heads, device, args.repeats
        )
        for L in args.lengths
    ]
    for r in rows:
        print(
            f"  L={r['seq_len']:>5}  r2_cpu={r['explicit_r2_cpu_s'] * 1e3:9.2f} ms  "
            f"r2_gpu={r.get('explicit_r2_gpu_s', float('nan')) * 1e3:8.2f} ms  "
            f"bias_epoch={r['ld_bias_epoch_s'] * 1e3:8.2f} ms  "
            f"overhead={r.get('ld_overhead_pct', float('nan')):6.1f}%"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "settings": vars(args),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] {out.resolve()}")


if __name__ == "__main__":
    main()
