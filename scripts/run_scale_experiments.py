"""Long-window and multi-chromosome experiments for the LD attention bias.

The reported 128-SNP benchmark cannot answer two questions a reviewer will ask
first, and both are listed as open in the manuscript's limitations:

``window``
    Does the layer still beat the explicit-r^2 pipeline when the window is long
    enough that sparse partner selection is actually load-bearing? At 128 sites
    a per-site regression can almost afford every other site, so the r^2 ranking
    -- the entire reason the matrix is built -- buys the control little. At 512
    or 1024 sites it cannot. This sweep also records the physical span and the
    peak GPU memory, since attention is quadratic in the number of loci.

``chromosome``
    Does a trained bias transfer to a chromosome it has never seen? This is the
    portability claim: the functional position encoding and the distance/context
    terms are functions of genomic coordinate rather than per-site lookups, so
    they *should* carry over, whereas an explicit-r^2 pipeline has to rebuild
    its matrix and refit per chromosome. The model is scored zero-shot on
    held-out chromosomes against controls that are refitted there.

Both experiments score every method on the same masked entries of the same
held-out individuals, so the per-seed differences stay paired. Rows are appended
as each run finishes, so an interrupted sweep still leaves usable results.

    .venv312/bin/python3.12 scripts/run_scale_experiments.py --experiment window
    .venv312/bin/python3.12 scripts/run_scale_experiments.py --experiment chromosome
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
    majority_baseline,
    select_ld_partners,
)
from ldattention.tasks.imputation import LDAwareImputationModel  # noqa: E402
from ldattention.validation import (  # noqa: E402
    RunConfig,
    _maf_stratified_accuracy,
    build_dataset,
    compute_true_r2,
    evaluate_model,
    extract_mean_attention,
    haplotypes_for,
    make_eval_masks,
    minor_allele_freq,
    partial_pearson,
    pearson,
    simulate_haplotypes_msprime,
    split_indices,
    subset,
    train_model,
    upper_offdiag,
)

# (use_distance_bias, use_genotype_bias)
ARMS = {"both": (True, True), "no_bias": (False, False)}

# Equal epoch budget across window lengths so length is not confounded with
# undertraining. Mixed precision + genotype-folded attention frees enough memory
# on a 6 GB card for these batches; attention remains O(L^2) in compute.
WINDOW_EPOCHS = {128: 300, 256: 300, 512: 300, 1024: 300}
WINDOW_BATCH = {128: 128, 256: 128, 512: 64, 1024: 48}


def _model(cfg: RunConfig, device: torch.device) -> LDAwareImputationModel:
    return LDAwareImputationModel(
        input_dim=3,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        genotype_rank=cfg.genotype_rank,
        max_distance=1.0,
        use_distance_bias=cfg.use_distance_bias,
        use_genotype_bias=cfg.use_genotype_bias,
        position_frequencies=cfg.position_frequencies,
    ).to(device)


def _base_config(n_sites: int, epochs: int, batch_size: int, **overrides) -> RunConfig:
    """The reported protocol, with only the window-dependent knobs changed."""
    cfg = RunConfig(
        name="tmp",
        n_sites=n_sites,
        n_haplotypes=2000,
        use_msprime=True,
        min_maf=0.05,
        hidden_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        genotype_rank=32,
        position_frequencies=16,
        epochs=epochs,
        batch_size=batch_size,
        lr=1.5e-3,
        mask_rate=0.3,
        mask_rate_jitter=0.2,
        block_missing=True,
        block_len=8,
        eval_every=10,
        n_eval_repeats=5,
        mask_rate_sweep=(),
        baseline_top_k=8,
        baseline_epochs=150,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _attention_corr(
    model, train_data, r2: np.ndarray, abs_dist: np.ndarray, n_sites: int, max_rows: int = 96
) -> tuple[float, float]:
    """Pearson and distance-controlled Pearson of mean attention against true r^2.

    Attention extraction materializes [B, H, L, L] weights, so the batch and the
    number of individuals are capped: the mean over individuals is already stable
    well before the full training split is consumed.
    """
    rows = min(max_rows, train_data["features"].shape[0])
    sub = {k: v[:rows] for k, v in train_data.items()}
    batch = max(1, min(8, int(2**23 / max(n_sites * n_sites, 1))))
    attn = extract_mean_attention(model, sub, batch, use_population=False)
    attn_flat, r2_flat = upper_offdiag(attn), upper_offdiag(r2)
    dist_flat = upper_offdiag(abs_dist)
    return pearson(attn_flat, r2_flat), partial_pearson(attn_flat, r2_flat, dist_flat)


def _fit_explicit_ld(
    r2_train: np.ndarray, top_k: int, cfg: RunConfig, train_data, generator, device
) -> LDRegressionBaseline:
    partners = select_ld_partners(r2_train, top_k)
    reg = LDRegressionBaseline(partners, device=device, epochs=cfg.baseline_epochs)
    reg.fit(
        train_data["features"], train_data["labels"], cfg.mask_rate, generator,
        batch_size=cfg.batch_size, block_missing=cfg.block_missing, block_len=cfg.block_len,
    )
    return reg


class RowWriter:
    """Append-as-you-go CSV, so a killed sweep still leaves complete rows."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path, self.columns = path, columns
        self.rows: list[dict] = []
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(
                {c: row.get(c) for c in self.columns}
            )


# --------------------------------------------------------------------------- #
# Experiment 1: window length
# --------------------------------------------------------------------------- #
WINDOW_COLUMNS = [
    "n_sites", "span_bp", "arm", "seed", "epochs", "batch_size", "best_epoch",
    "accuracy", "dosage_r2", "majority", "explicit_ld_top8", "explicit_ld_top64",
    "model_minus_top8", "model_minus_top64",
    "accuracy_maf_low", "accuracy_maf_mid", "accuracy_maf_high",
    "attention_vs_r2_pearson", "attention_vs_r2_partial_pearson",
    "train_seconds", "peak_gb",
]


def run_window(args, device: torch.device, out_dir: Path) -> None:
    writer = RowWriter(out_dir / "window_raw.csv", WINDOW_COLUMNS)
    windows = [int(x) for x in args.windows.split(",") if x]

    for n_sites in windows:
        epochs = args.epochs or WINDOW_EPOCHS.get(n_sites, 150)
        batch_size = WINDOW_BATCH.get(n_sites, 16)

        for seed in range(args.n_seeds):
            cfg = _base_config(n_sites, epochs, batch_size)
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)

            hap, positions, span_bp = simulate_haplotypes_msprime(
                cfg.n_haplotypes, n_sites, seed, mutation_rate=2.5e-8,
                min_maf=cfg.min_maf, return_span_bp=True,
            )
            data = build_dataset(hap, positions, np.zeros(hap.shape[0], dtype=np.int64), device)
            n_ind = data["features"].shape[0]
            train_idx, val_idx, test_idx = split_indices(n_ind, cfg.val_frac, cfg.test_frac, rng)
            train_data, val_data, test_data = (subset(data, i) for i in (train_idx, val_idx, test_idx))

            train_haps = haplotypes_for(hap[: n_ind * 2], train_idx)
            r2_train = compute_true_r2(train_haps)
            maf = minor_allele_freq(train_haps)
            abs_dist = np.abs(positions[:, None] - positions[None, :])

            val_masks = make_eval_masks(
                val_data["features"].shape[0], n_sites, cfg.mask_rate, 2, seed + 991,
                device, cfg.block_missing, cfg.block_len,
            )
            test_masks = make_eval_masks(
                test_data["features"].shape[0], n_sites, cfg.mask_rate, cfg.n_eval_repeats,
                seed + 7717, device, cfg.block_missing, cfg.block_len,
            )

            # Controls do not depend on the model, so they are fit once per
            # (window, seed) and scored against both arms on identical entries.
            majority = majority_baseline(train_data["labels"], test_data["labels"], test_masks)
            explicit = {}
            for top_k in (8, min(64, n_sites - 1)):
                reg = _fit_explicit_ld(r2_train, top_k, cfg, train_data, generator, device)
                explicit[top_k] = reg.score(test_data["features"], test_data["labels"], test_masks)
                del reg
                torch.cuda.empty_cache()

            for arm, (dist_bias, geno_bias) in ARMS.items():
                cfg.name, cfg.use_distance_bias, cfg.use_genotype_bias = arm, dist_bias, geno_bias
                torch.manual_seed(seed)
                gen = torch.Generator(device=device).manual_seed(seed)
                model = _model(cfg, device)

                torch.cuda.reset_peak_memory_stats()
                started = time.time()
                fit = train_model(
                    model, train_data, cfg, gen, use_population=False,
                    val_data=val_data, val_masks=val_masks,
                )
                train_seconds = time.time() - started
                peak_gb = torch.cuda.max_memory_allocated() / 2**30

                ev = evaluate_model(model, test_data, test_masks, cfg, use_population=False)
                attn_r, attn_partial = _attention_corr(model, train_data, r2_train, abs_dist, n_sites)

                top64 = min(64, n_sites - 1)
                row = {
                    "n_sites": n_sites, "span_bp": round(span_bp), "arm": arm, "seed": seed,
                    "epochs": epochs, "batch_size": batch_size, "best_epoch": fit["best_epoch"],
                    "accuracy": ev["accuracy"], "dosage_r2": ev["dosage_r2"],
                    "majority": majority["accuracy"],
                    "explicit_ld_top8": explicit[8]["accuracy"],
                    "explicit_ld_top64": explicit[top64]["accuracy"],
                    "model_minus_top8": ev["accuracy"] - explicit[8]["accuracy"],
                    "model_minus_top64": ev["accuracy"] - explicit[top64]["accuracy"],
                    **_maf_stratified_accuracy(ev["correct"], ev["sites"], maf),
                    "attention_vs_r2_pearson": attn_r,
                    "attention_vs_r2_partial_pearson": attn_partial,
                    "train_seconds": round(train_seconds, 1), "peak_gb": round(peak_gb, 2),
                }
                writer.add(row)
                print(
                    f"  [window] L={n_sites:5d} {arm:8s} seed={seed} "
                    f"acc={row['accuracy']:.4f} top8={row['explicit_ld_top8']:.4f} "
                    f"top{top64}={row['explicit_ld_top64']:.4f} "
                    f"d8={row['model_minus_top8']:+.4f} d{top64}={row['model_minus_top64']:+.4f} "
                    f"attn~r2={attn_r:+.3f} peak={peak_gb:.1f}GB ({train_seconds:.0f}s)",
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()

            del data, train_data, val_data, test_data
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Experiment 2: multiple chromosomes and zero-shot transfer
# --------------------------------------------------------------------------- #
CHROM_COLUMNS = [
    "arm", "seed", "chrom", "kind", "n_sites", "span_bp", "n_train_chrom", "pos_freq", "best_epoch",
    "accuracy", "dosage_r2", "majority", "explicit_ld", "model_minus_explicit_ld",
    "attention_vs_r2_pearson", "attention_vs_r2_partial_pearson", "train_seconds",
]


def run_chromosome(args, device: torch.device, out_dir: Path) -> None:
    writer = RowWriter(out_dir / "chrom_raw.csv", CHROM_COLUMNS)
    n_sites = args.chrom_sites
    epochs = args.chrom_epochs
    batch_size = WINDOW_BATCH.get(n_sites, 32)
    n_chrom, n_train_chrom = args.n_chrom, args.n_train_chrom

    for seed in range(args.n_seeds):
        cfg = _base_config(n_sites, epochs, batch_size, position_frequencies=args.pos_freq)
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Independent coalescent runs are independent chromosomes. The same
        # individuals are genotyped on all of them, and the split is shared, so
        # a test individual is unseen on every chromosome.
        chroms = []
        for c in range(n_chrom):
            hap, positions, span_bp = simulate_haplotypes_msprime(
                cfg.n_haplotypes, n_sites, seed * 100 + c, mutation_rate=2.5e-8,
                min_maf=cfg.min_maf, return_span_bp=True,
            )
            data = build_dataset(hap, positions, np.zeros(hap.shape[0], dtype=np.int64), device)
            chroms.append({"hap": hap, "positions": positions, "span_bp": span_bp, "data": data})

        n_ind = chroms[0]["data"]["features"].shape[0]
        train_idx, val_idx, test_idx = split_indices(n_ind, cfg.val_frac, cfg.test_frac, rng)

        for ch in chroms:
            ch["train"] = subset(ch["data"], train_idx)
            ch["val"] = subset(ch["data"], val_idx)
            ch["test"] = subset(ch["data"], test_idx)
            train_haps = haplotypes_for(ch["hap"][: n_ind * 2], train_idx)
            ch["r2"] = compute_true_r2(train_haps)
            ch["maf"] = minor_allele_freq(train_haps)
            ch["abs_dist"] = np.abs(ch["positions"][:, None] - ch["positions"][None, :])

        # Identical mask draws on every chromosome keep the comparison paired.
        test_masks = make_eval_masks(
            len(test_idx), n_sites, cfg.mask_rate, cfg.n_eval_repeats, seed + 7717,
            device, cfg.block_missing, cfg.block_len,
        )

        # The explicit-LD pipeline is rebuilt and refitted on every chromosome --
        # that is exactly the per-cohort cost the learned bias is meant to avoid.
        for c, ch in enumerate(chroms):
            ch["majority"] = majority_baseline(ch["train"]["labels"], ch["test"]["labels"], test_masks)
            reg = _fit_explicit_ld(ch["r2"], cfg.baseline_top_k, cfg, ch["train"], generator, device)
            ch["explicit"] = reg.score(ch["test"]["features"], ch["test"]["labels"], test_masks)
            del reg
            torch.cuda.empty_cache()
            print(
                f"  [chrom] seed={seed} chr{c} controls: majority={ch['majority']['accuracy']:.4f} "
                f"explicitLD={ch['explicit']['accuracy']:.4f}",
                flush=True,
            )

        runs = [
            ("both_multi", True, True, n_train_chrom),
            ("no_bias_multi", False, False, n_train_chrom),
            # One chromosome of training data isolates whether transfer comes
            # from the bias itself or merely from seeing several chromosomes.
            ("both_single", True, True, 1),
        ]
        if args.arms:
            keep = set(args.arms.split(","))
            runs = [r for r in runs if r[0] in keep]

        for arm, dist_bias, geno_bias, k_train in runs:
            cfg.name, cfg.use_distance_bias, cfg.use_genotype_bias = arm, dist_bias, geno_bias
            torch.manual_seed(seed)
            gen = torch.Generator(device=device).manual_seed(seed)

            pooled_train = {
                key: torch.cat([ch["train"][key] for ch in chroms[:k_train]], dim=0)
                for key in chroms[0]["train"]
            }
            pooled_val = {
                key: torch.cat([ch["val"][key] for ch in chroms[:k_train]], dim=0)
                for key in chroms[0]["val"]
            }
            val_masks = make_eval_masks(
                pooled_val["features"].shape[0], n_sites, cfg.mask_rate, 2, seed + 991,
                device, cfg.block_missing, cfg.block_len,
            )

            model = _model(cfg, device)
            started = time.time()
            fit = train_model(
                model, pooled_train, cfg, gen, use_population=False,
                val_data=pooled_val, val_masks=val_masks,
            )
            train_seconds = time.time() - started
            del pooled_train, pooled_val
            torch.cuda.empty_cache()

            for c, ch in enumerate(chroms):
                ev = evaluate_model(model, ch["test"], test_masks, cfg, use_population=False)
                attn_r, attn_partial = _attention_corr(
                    model, ch["train"], ch["r2"], ch["abs_dist"], n_sites
                )
                kind = "within" if c < k_train else "cross"
                row = {
                    "arm": arm, "seed": seed, "chrom": c, "kind": kind, "n_sites": n_sites,
                    "span_bp": round(ch["span_bp"]), "n_train_chrom": k_train,
                    "pos_freq": cfg.position_frequencies,
                    "best_epoch": fit["best_epoch"], "accuracy": ev["accuracy"],
                    "dosage_r2": ev["dosage_r2"], "majority": ch["majority"]["accuracy"],
                    "explicit_ld": ch["explicit"]["accuracy"],
                    "model_minus_explicit_ld": ev["accuracy"] - ch["explicit"]["accuracy"],
                    "attention_vs_r2_pearson": attn_r,
                    "attention_vs_r2_partial_pearson": attn_partial,
                    "train_seconds": round(train_seconds, 1),
                }
                writer.add(row)
                print(
                    f"  [chrom] {arm:14s} seed={seed} chr{c} {kind:6s} "
                    f"acc={ev['accuracy']:.4f} explicitLD={ch['explicit']['accuracy']:.4f} "
                    f"delta={row['model_minus_explicit_ld']:+.4f} attn~r2={attn_r:+.3f}",
                    flush=True,
                )

            del model
            torch.cuda.empty_cache()

        for ch in chroms:
            ch.clear()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[setup] device={device.type} ({device_name}) experiment={args.experiment} "
          f"seeds={args.n_seeds} out_dir={out_dir.resolve()}", flush=True)

    started = time.time()
    if args.experiment in ("window", "both"):
        run_window(args, device, out_dir)
    if args.experiment in ("chromosome", "both"):
        run_chromosome(args, device, out_dir)

    (out_dir / "config.json").write_text(json.dumps({
        "experiment": args.experiment,
        "device": device.type,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "n_seeds": args.n_seeds,
        "windows": args.windows,
        "window_epochs": WINDOW_EPOCHS,
        "window_batch": WINDOW_BATCH,
        "use_amp": True,
        "chrom_sites": args.chrom_sites,
        "chrom_epochs": args.chrom_epochs,
        "n_chrom": args.n_chrom,
        "n_train_chrom": args.n_train_chrom,
        "total_seconds": time.time() - started,
    }, indent=2))
    print(f"\n[done] {time.time() - started:.0f}s; results in {out_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("window", "chromosome", "both"), default="both")
    p.add_argument("--out_dir", type=str, default="results_scale")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--windows", type=str, default="128,256,512,1024")
    p.add_argument("--epochs", type=int, default=0, help="0 = per-window default")
    p.add_argument("--chrom_sites", type=int, default=256)
    p.add_argument("--chrom_epochs", type=int, default=150)
    p.add_argument("--n_chrom", type=int, default=6)
    p.add_argument("--n_train_chrom", type=int, default=4)
    # Cross-chromosome transfer collapses with the default 16 sinusoidal
    # frequencies; lowering this trades site resolution for portability.
    p.add_argument("--pos_freq", type=int, default=16)
    p.add_argument("--arms", type=str, default="", help="comma-separated subset of arms to run")
    return p.parse_args()


if __name__ == "__main__":
    main()
