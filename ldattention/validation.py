"""Reusable validation utilities for LD-aware attention.

This module contains the building blocks used by both ``scripts/validate_ld.py``
(single run) and ``scripts/run_experiments.py`` (ablation + multi-seed sweep):

- haplotype simulation with known recombination structure (Li--Stephens copying
  model, single- or multi-population; optional msprime),
- ground-truth pairwise LD (``r^2``),
- masked-genotype imputation training,
- learned-attention / learned-bias extraction,
- scipy-free Pearson/Spearman correlations,
- a single ``run_config`` entry point returning a results dict.

Nothing here ever feeds an explicit LD matrix into the model; ``r^2`` is used for
*evaluation only*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ldattention.models.ld_bias import LDAttentionBias
from ldattention.tasks.imputation import LDAwareImputationModel


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """One experiment configuration (a point in the ablation grid)."""

    name: str = "both"
    use_distance_bias: bool = True
    use_genotype_bias: bool = True
    pop_film: bool = False  # whether the *model* uses per-population FiLM conditioning
    # Simulation
    n_haplotypes: int = 600
    n_sites: int = 48
    n_founders: int = 5
    rho: float = 15.0
    mutation_rate: float = 0.005
    n_populations: int = 1  # number of populations in the *data*
    use_msprime: bool = False
    msprime_mutation_rate: float = 2.5e-8
    # Genotyping arrays and imputation panels carry common variants; without a
    # MAF floor the coalescent SFS is dominated by singletons and "predict the
    # major allele" already scores ~0.90, which hides all real signal.
    min_maf: float = 0.05
    # Model / training
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.0
    genotype_rank: int = 16
    # 0 keeps the raw position scalar; >0 expands it into that many sin/cos
    # frequency pairs so the shared input projection can resolve individual sites.
    position_frequencies: int = 0
    epochs: int = 200
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    mask_rate: float = 0.2
    # Contiguous dropout, like a GBS lane dropping a stretch of the window,
    # instead of hiding sites independently. Nearby LD partners vanish together,
    # which is exactly the regime where a learned bias over all sites should
    # beat a regression on a pre-chosen top-k list.
    block_missing: bool = False
    block_len: int = 8
    label_smoothing: float = 0.0
    use_missing_channel: bool = True
    # Sampling the per-batch mask rate from a range teaches one model to serve
    # every missingness level instead of the single rate it was trained at.
    mask_rate_jitter: float = 0.0
    # Held-out evaluation
    val_frac: float = 0.15
    test_frac: float = 0.20
    eval_mask_rate: float | None = None
    n_eval_repeats: int = 5
    eval_every: int = 5
    # Extra missingness levels to score the *same* trained model at. Each level
    # gets its own freshly fitted explicit-LD control, so the comparison stays
    # paired at every point on the curve.
    mask_rate_sweep: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7)
    # Explicit-LD control
    baseline_top_k: int = 8
    baseline_epochs: int = 120

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_haplotypes_copying(
    n_haplotypes: int,
    n_sites: int,
    n_founders: int,
    rho: float,
    mutation_rate: float,
    rng: np.random.Generator,
    positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Li--Stephens style mosaic-copying simulation for one population.

    Returns ``(haplotypes [n_haplotypes, n_sites] in {0,1}, positions [n_sites])``.
    A shared ``positions`` array can be passed so multiple populations align on
    the same variant coordinates.
    """
    if positions is None:
        positions = np.sort(rng.uniform(0.0, 1.0, size=n_sites)).astype(np.float32)
    site_freq = rng.uniform(0.1, 0.9, size=n_sites)
    founders = (rng.uniform(size=(n_founders, n_sites)) < site_freq).astype(np.int8)

    gaps = np.diff(positions)
    switch_prob = 1.0 - np.exp(-rho * gaps)

    haplotypes = np.empty((n_haplotypes, n_sites), dtype=np.int8)
    for h in range(n_haplotypes):
        cur = rng.integers(n_founders)
        for i in range(n_sites):
            if i > 0 and rng.uniform() < switch_prob[i - 1]:
                cur = rng.integers(n_founders)
            haplotypes[h, i] = founders[cur, i]
    if mutation_rate > 0:
        flip = rng.uniform(size=haplotypes.shape) < mutation_rate
        haplotypes = np.where(flip, 1 - haplotypes, haplotypes).astype(np.int8)
    return haplotypes, positions.astype(np.float32)


def simulate_haplotypes_msprime(
    n_haplotypes: int,
    n_sites: int,
    seed: int,
    mutation_rate: float = 2.5e-8,
    min_maf: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Coalescent-with-recombination simulation using msprime (optional)."""
    import msprime  # noqa: PLC0415

    # Longer requested windows need a longer chromosome so MAF filtering still
    # leaves a contiguous block of common SNPs.
    seq_len = 1_000_000 if n_sites <= 64 else 2_500_000
    # msprime requires strictly positive seeds.
    ts = msprime.sim_ancestry(
        samples=max(n_haplotypes // 2, 1),  # diploid individuals -> ~n_haplotypes genomes
        recombination_rate=1e-8,
        sequence_length=seq_len,
        population_size=10_000,
        random_seed=seed + 1,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed + 2)
    geno = ts.genotype_matrix()
    pos = np.array([v.site.position for v in ts.variants()], dtype=np.float32)
    biallelic = geno.max(axis=1) <= 1
    geno, pos = geno[biallelic], pos[biallelic]
    if min_maf > 0:
        freq = geno.mean(axis=1)
        common = np.minimum(freq, 1.0 - freq) >= min_maf
        geno, pos = geno[common], pos[common]
    if geno.shape[0] < n_sites:
        raise RuntimeError("msprime produced fewer variants than requested")
    # Take a *contiguous* central window of variants. Physically-adjacent SNPs
    # carry strong LD (this is the dense-window regime real imputation operates
    # in), rather than spreading sites thinly across the whole sequence.
    start = max((geno.shape[0] - n_sites) // 2, 0)
    idx = np.arange(start, start + n_sites)
    geno, pos = geno[idx], pos[idx]
    positions = (pos - pos.min()) / (pos.max() - pos.min() + 1e-9)
    return geno.T.astype(np.int8), positions.astype(np.float32)


def simulate(cfg: RunConfig, rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Simulate one dataset (single or multi-population).

    Returns a dict with keys: ``haplotypes`` [N, L], ``positions`` [L],
    ``pop_labels`` [N] (population index per haplotype), and ``r2_per_pop`` (list
    of per-population r^2 matrices).
    """
    if cfg.use_msprime:
        haplotypes, positions = simulate_haplotypes_msprime(
            cfg.n_haplotypes, cfg.n_sites, seed,
            mutation_rate=cfg.msprime_mutation_rate, min_maf=cfg.min_maf,
        )
        pop_labels = np.zeros(haplotypes.shape[0], dtype=np.int64)
        return {
            "haplotypes": haplotypes,
            "positions": positions,
            "pop_labels": pop_labels,
            "r2_per_pop": [compute_true_r2(haplotypes)],
        }

    positions = np.sort(rng.uniform(0.0, 1.0, size=cfg.n_sites)).astype(np.float32)
    per_pop = max(cfg.n_populations, 1)
    hap_per_pop = cfg.n_haplotypes // per_pop
    hap_chunks, label_chunks, r2_per_pop = [], [], []
    for p in range(per_pop):
        hap, _ = simulate_haplotypes_copying(
            hap_per_pop, cfg.n_sites, cfg.n_founders, cfg.rho, cfg.mutation_rate, rng, positions=positions
        )
        hap_chunks.append(hap)
        label_chunks.append(np.full(hap_per_pop, p, dtype=np.int64))
        r2_per_pop.append(compute_true_r2(hap))
    return {
        "haplotypes": np.concatenate(hap_chunks, axis=0),
        "positions": positions,
        "pop_labels": np.concatenate(label_chunks, axis=0),
        "r2_per_pop": r2_per_pop,
    }


# --------------------------------------------------------------------------- #
# Ground-truth LD
# --------------------------------------------------------------------------- #
def compute_true_r2(haplotypes: np.ndarray) -> np.ndarray:
    """Empirical pairwise LD (r^2) from phased haplotypes ``[n_hap, L]``."""
    h = haplotypes.astype(np.float64)
    std = h.std(axis=0)
    polymorphic = std > 1e-8
    corr = np.zeros((h.shape[1], h.shape[1]), dtype=np.float64)
    if polymorphic.sum() > 1:
        sub = h[:, polymorphic]
        c = np.corrcoef(sub, rowvar=False)
        idx = np.where(polymorphic)[0]
        corr[np.ix_(idx, idx)] = np.nan_to_num(c)
    r2 = corr**2
    np.fill_diagonal(r2, 1.0)
    return r2


# --------------------------------------------------------------------------- #
# Correlations (no scipy dependency)
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float((a @ b) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rankdata(a), _rankdata(b))


def upper_offdiag(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu]


def residualize_by(values: np.ndarray, control: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Remove the (arbitrary, non-parametric) effect of ``control`` from ``values``.

    Bins pairs by their ``control`` value (e.g. genomic distance) into quantile
    bins and subtracts the per-bin mean. This strips out *any* monotone or
    non-linear distance effect, so the residual isolates structure not explained
    by distance.
    """
    resid = values.astype(np.float64).copy()
    edges = np.quantile(control, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(control, edges[1:-1]), 0, n_bins - 1)
    for b in np.unique(bin_idx):
        m = bin_idx == b
        resid[m] = resid[m] - resid[m].mean()
    return resid


def partial_pearson(a: np.ndarray, b: np.ndarray, control: np.ndarray, n_bins: int = 16) -> float:
    """Pearson correlation of ``a`` and ``b`` after controlling for ``control``.

    Answers: does the model track LD *beyond* what genomic distance explains?
    """
    return pearson(residualize_by(a, control, n_bins), residualize_by(b, control, n_bins))


# --------------------------------------------------------------------------- #
# Dataset / training / extraction
# --------------------------------------------------------------------------- #
def build_dataset(
    haplotypes: np.ndarray,
    positions: np.ndarray,
    pop_labels: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pair haplotypes into diploid individuals (within population)."""
    n_ind = haplotypes.shape[0] // 2
    hap = haplotypes[: n_ind * 2]
    labels_pop = pop_labels[: n_ind * 2]
    a1 = hap[0::2].astype(np.float32)
    a2 = hap[1::2].astype(np.float32)
    features = np.stack([a1, a2], axis=-1)
    labels = (a1 + a2).astype(np.int64)
    ind_pop = labels_pop[0::2]  # both haplotypes share a population by construction
    pos = np.broadcast_to(positions[None, :, None], (n_ind, positions.shape[0], 1))
    return {
        "features": torch.tensor(features, device=device),
        "positions": torch.tensor(np.ascontiguousarray(pos), dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, device=device),
        "pop": torch.tensor(np.ascontiguousarray(ind_pop), dtype=torch.long, device=device),
    }


def split_indices(
    n: int, val_frac: float, test_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle individuals into disjoint train / validation / test index sets."""
    perm = rng.permutation(n)
    n_test = max(int(round(n * test_frac)), 1)
    n_val = max(int(round(n * val_frac)), 1)
    return perm[n_test + n_val :], perm[n_test : n_test + n_val], perm[:n_test]


def subset(data: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    sel = torch.as_tensor(idx, dtype=torch.long, device=data["features"].device)
    return {k: v[sel] for k, v in data.items()}


def haplotypes_for(haplotypes: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The two haplotype rows backing each selected individual."""
    rows = np.concatenate([2 * idx, 2 * idx + 1])
    return haplotypes[np.sort(rows)]


def minor_allele_freq(haplotypes: np.ndarray) -> np.ndarray:
    freq = haplotypes.astype(np.float64).mean(axis=0)
    return np.minimum(freq, 1.0 - freq)


MAF_BINS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.10),
    ("mid", 0.10, 0.25),
    ("high", 0.25, 0.51),
)


def sample_mask(
    n: int,
    n_sites: int,
    mask_rate: float,
    generator: torch.Generator,
    device: torch.device,
    block: bool = False,
    block_len: int = 8,
) -> torch.Tensor:
    """Boolean ``[n, n_sites]`` mask of hidden genotypes.

    Independent Bernoulli hides are the usual self-supervised default. Real GBS
    dropouts are stretches of the window, so ``block=True`` places overlapping
    contiguous blocks whose expected coverage is ``mask_rate``.
    """
    if not block:
        return torch.rand(n, n_sites, generator=generator, device=device) < mask_rate
    length = max(2, min(int(block_len), n_sites))
    # 1 - (1 - length/L)^k ≈ rate  =>  k = log(1-rate) / log(1 - length/L)
    frac = length / n_sites
    if frac >= 1.0:
        return torch.ones(n, n_sites, dtype=torch.bool, device=device)
    n_blocks = max(1, int(round(math.log(max(1.0 - mask_rate, 1e-6)) / math.log(1.0 - frac))))
    starts = torch.randint(0, n_sites, (n, n_blocks), generator=generator, device=device)
    pos = torch.arange(n_sites, device=device).view(1, 1, n_sites)
    return ((pos - starts.unsqueeze(-1)) % n_sites < length).any(dim=1)


def _prepare_masked(
    features2: torch.Tensor,
    mask_rate: float,
    generator: torch.Generator,
    use_missing_channel: bool,
    block_missing: bool = False,
    block_len: int = 8,
):
    """Hide sites and build the model input (optionally with a
    missingness-indicator channel so the model *knows* which sites are missing)."""
    b, l, _ = features2.shape
    mask = sample_mask(b, l, mask_rate, generator, features2.device, block_missing, block_len)
    masked = features2.clone()
    if use_missing_channel:
        masked[mask] = 0.0
        missing = mask.to(masked.dtype).unsqueeze(-1)
        feats = torch.cat([masked, missing], dim=-1)  # [B, L, 3]
    else:
        masked[mask] = -1.0  # ambiguous sentinel (no explicit missingness signal)
        feats = masked
    return feats, mask


def _prepare_eval(features2: torch.Tensor, use_missing_channel: bool) -> torch.Tensor:
    """Model input for evaluation (nothing masked -> missingness channel all zeros)."""
    if not use_missing_channel:
        return features2
    b, l, _ = features2.shape
    zeros = torch.zeros(b, l, 1, dtype=features2.dtype, device=features2.device)
    return torch.cat([features2, zeros], dim=-1)


def make_eval_masks(
    n: int, n_sites: int, mask_rate: float, n_repeats: int, seed: int, device: torch.device,
    block_missing: bool = False, block_len: int = 8,
) -> list[torch.Tensor]:
    """Fixed masking draws, shared by the model and every baseline."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return [
        sample_mask(n, n_sites, mask_rate, generator, device, block_missing, block_len)
        for _ in range(n_repeats)
    ]


@torch.no_grad()
def evaluate_model(
    model: LDAwareImputationModel,
    data: dict[str, torch.Tensor],
    masks: list[torch.Tensor],
    cfg: RunConfig,
    use_population: bool,
) -> dict[str, Any]:
    """Score held-out individuals on pre-drawn masks.

    Returns overall accuracy, squared dosage correlation (the usual imputation-
    quality metric), and per-masked-entry correctness with its site index so
    accuracy can be stratified by allele frequency afterwards.
    """
    model.eval()
    features, positions, labels, pop = data["features"], data["positions"], data["labels"], data["pop"]
    n = features.shape[0]
    correct_flags: list[torch.Tensor] = []
    site_ids: list[torch.Tensor] = []
    dosage_pred: list[torch.Tensor] = []
    dosage_true: list[torch.Tensor] = []
    class_values = torch.arange(3, dtype=torch.float32, device=features.device)

    for mask in masks:
        for start in range(0, n, cfg.batch_size):
            stop = start + cfg.batch_size
            feats2, poss, labs = features[start:stop], positions[start:stop], labels[start:stop]
            sub_mask = mask[start:stop]
            if not bool(sub_mask.any()):
                continue
            masked = feats2.clone()
            masked[sub_mask] = 0.0
            if cfg.use_missing_channel:
                feats = torch.cat([masked, sub_mask.to(masked.dtype).unsqueeze(-1)], dim=-1)
            else:
                masked[sub_mask] = -1.0
                feats = masked
            pop_id = pop[start:stop] if use_population else None
            logits, _ = model(feats, poss, population_id=pop_id)
            probs = logits.softmax(dim=-1)
            correct_flags.append((logits.argmax(-1)[sub_mask] == labs[sub_mask]).to(torch.float32))
            site_ids.append(sub_mask.nonzero(as_tuple=True)[1])
            dosage_pred.append((probs * class_values).sum(-1)[sub_mask])
            dosage_true.append(labs[sub_mask].to(torch.float32))

    flags = torch.cat(correct_flags).cpu().numpy()
    sites = torch.cat(site_ids).cpu().numpy()
    pred = torch.cat(dosage_pred).cpu().numpy()
    true = torch.cat(dosage_true).cpu().numpy()
    return {
        "accuracy": float(flags.mean()) if flags.size else 0.0,
        "dosage_r2": float(pearson(pred, true) ** 2) if flags.size else 0.0,
        "correct": flags,
        "sites": sites,
    }


def train_model(
    model: LDAwareImputationModel,
    data: dict[str, torch.Tensor],
    cfg: RunConfig,
    generator: torch.Generator,
    use_population: bool,
    val_data: dict[str, torch.Tensor] | None = None,
    val_masks: list[torch.Tensor] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train on the training split, keeping the checkpoint that scores best on validation.

    Returns the final training accuracy, the best validation accuracy, and the
    epoch it came from. Model weights are restored to that best checkpoint.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    features, positions, labels, pop = data["features"], data["positions"], data["labels"], data["pop"]
    n = features.shape[0]
    steps_per_epoch = max(math.ceil(n / cfg.batch_size), 1)
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(int(total_steps * cfg.warmup_frac), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    do_val = val_data is not None and val_masks is not None

    best_val = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    train_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=generator, device=features.device)
        correct = total = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            feats2, poss, labs = features[idx], positions[idx], labels[idx]
            pop_id = pop[idx] if use_population else None
            rate = cfg.mask_rate
            if cfg.mask_rate_jitter > 0:
                low = max(cfg.mask_rate - cfg.mask_rate_jitter, 0.02)
                high = min(cfg.mask_rate + cfg.mask_rate_jitter, 0.9)
                draw = torch.rand(1, generator=generator, device=features.device).item()
                rate = low + draw * (high - low)
            masked_feats, mask = _prepare_masked(
                feats2, rate, generator, cfg.use_missing_channel,
                cfg.block_missing, cfg.block_len,
            )
            if not bool(mask.any()):
                continue
            logits, _ = model(masked_feats, poss, population_id=pop_id)
            loss = F.cross_entropy(logits[mask], labs[mask], label_smoothing=cfg.label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                correct += int((logits[mask].argmax(-1) == labs[mask]).sum())
                total += int(mask.sum())
        train_acc = correct / max(total, 1)

        if do_val and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            val_acc = evaluate_model(model, val_data, val_masks, cfg, use_population)["accuracy"]
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    epoch={epoch:4d} train={train_acc:.4f} val={val_acc:.4f}")
        elif verbose and (epoch % max(cfg.epochs // 5, 1) == 0 or epoch == 1):
            print(f"    epoch={epoch:4d} train={train_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"train_accuracy": train_acc, "val_accuracy": max(best_val, 0.0), "best_epoch": best_epoch}


@torch.no_grad()
def extract_mean_attention(model, data, batch_size, use_population, pop_filter=None, use_missing_channel=True) -> np.ndarray:
    model.eval()
    features, positions, pop = data["features"], data["positions"], data["pop"]
    sel = slice(None) if pop_filter is None else (pop == pop_filter)
    features, positions, pop = features[sel], positions[sel], pop[sel]
    n, l, _ = features.shape
    attn_sum = torch.zeros(l, l, device=features.device)
    count = 0
    for start in range(0, n, batch_size):
        feats = _prepare_eval(features[start : start + batch_size], use_missing_channel)
        poss = positions[start : start + batch_size]
        pop_id = pop[start : start + batch_size] if use_population else None
        _, attns = model(feats, poss, population_id=pop_id, need_weights=True)
        for a in attns:
            attn_sum += a.mean(dim=(0, 1))
            count += 1
    attn = (attn_sum / max(count, 1)).cpu().numpy()
    return 0.5 * (attn + attn.T)


@torch.no_grad()
def extract_mean_bias(model, data, batch_size, use_population, pop_filter=None, use_missing_channel=True) -> np.ndarray:
    model.eval()
    features, positions, pop = data["features"], data["positions"], data["pop"]
    sel = slice(None) if pop_filter is None else (pop == pop_filter)
    features, positions, pop = features[sel], positions[sel], pop[sel]
    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out.detach())

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, LDAttentionBias)]
    try:
        n, l, _ = features.shape
        bias_sum = torch.zeros(l, l, device=features.device)
        count = 0
        for start in range(0, n, batch_size):
            captured.clear()
            feats = _prepare_eval(features[start : start + batch_size], use_missing_channel)
            poss = positions[start : start + batch_size]
            pop_id = pop[start : start + batch_size] if use_population else None
            model(feats, poss, population_id=pop_id, need_weights=False)
            for b in captured:
                bias_sum += b.mean(dim=(0, 1))
                count += 1
        bias = (bias_sum / max(count, 1)).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return 0.5 * (bias + bias.T)


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def _maf_stratified_accuracy(
    correct: np.ndarray, sites: np.ndarray, maf: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    """Accuracy within each allele-frequency band (empty bands report NaN)."""
    out: dict[str, float] = {}
    site_maf = maf[sites]
    for label, low, high in MAF_BINS:
        sel = (site_maf >= low) & (site_maf < high)
        out[f"{prefix}accuracy_maf_{label}"] = float(correct[sel].mean()) if sel.any() else float("nan")
    return out


def run_config(cfg: RunConfig, seed: int, device: torch.device, verbose: bool = False) -> dict[str, Any]:
    """Run one full simulate -> split -> train -> held-out evaluate cycle.

    Every reported accuracy comes from test individuals the model never saw, on
    masking draws shared with the baselines. Ground-truth ``r^2`` is derived from
    training haplotypes only and is used for evaluation and for the explicit-LD
    control -- it is never an input to the model.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    multi_pop_data = cfg.n_populations > 1
    use_pop_model = cfg.pop_film  # model conditions on population only if requested

    sim = simulate(cfg, rng, seed)
    data = build_dataset(sim["haplotypes"], sim["positions"], sim["pop_labels"], device)

    n_individuals = data["features"].shape[0]
    train_idx, val_idx, test_idx = split_indices(n_individuals, cfg.val_frac, cfg.test_frac, rng)
    train_data, val_data, test_data = (subset(data, i) for i in (train_idx, val_idx, test_idx))

    # LD is estimated on training haplotypes only, so the explicit-LD control
    # gets no more information than the model does.
    haplotypes = sim["haplotypes"][: n_individuals * 2]
    train_haps = haplotypes_for(haplotypes, train_idx)
    maf = minor_allele_freq(train_haps)

    eval_rate = cfg.eval_mask_rate if cfg.eval_mask_rate is not None else cfg.mask_rate
    n_sites = data["features"].shape[1]
    val_masks = make_eval_masks(
        val_data["features"].shape[0], n_sites, eval_rate, 2, seed + 991, device,
        cfg.block_missing, cfg.block_len,
    )
    test_masks = make_eval_masks(
        test_data["features"].shape[0], n_sites, eval_rate, cfg.n_eval_repeats, seed + 7717, device,
        cfg.block_missing, cfg.block_len,
    )

    model = LDAwareImputationModel(
        input_dim=3 if cfg.use_missing_channel else 2,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        genotype_rank=cfg.genotype_rank,
        max_distance=1.0,
        use_distance_bias=cfg.use_distance_bias,
        use_genotype_bias=cfg.use_genotype_bias,
        num_populations=cfg.n_populations if use_pop_model else 0,
        position_frequencies=cfg.position_frequencies,
    ).to(device)

    fit = train_model(
        model, train_data, cfg, generator, use_pop_model,
        val_data=val_data, val_masks=val_masks, verbose=verbose,
    )
    test_eval = evaluate_model(model, test_data, test_masks, cfg, use_pop_model)

    # ---- baselines on the identical masked entries ---------------------------
    from ldattention.baselines import (  # noqa: PLC0415
        LDRegressionBaseline,
        majority_baseline,
        select_ld_partners,
        time_explicit_r2,
    )

    r2_train = compute_true_r2(train_haps)
    r2_seconds = time_explicit_r2(train_haps)
    partner_idx = select_ld_partners(r2_train, cfg.baseline_top_k)

    def fit_ld_regression(rate: float) -> LDRegressionBaseline:
        reg = LDRegressionBaseline(partner_idx, device=device, epochs=cfg.baseline_epochs)
        reg.fit(
            train_data["features"], train_data["labels"], rate, generator, batch_size=cfg.batch_size,
            block_missing=cfg.block_missing, block_len=cfg.block_len,
        )
        return reg

    majority = majority_baseline(train_data["labels"], test_data["labels"], test_masks)
    ld_reg = fit_ld_regression(eval_rate)
    ld_reg_eval = ld_reg.score(test_data["features"], test_data["labels"], test_masks)

    # ---- accuracy as a function of missingness -------------------------------
    # The single trained model is reused at every level; the explicit-LD control
    # is refitted at each level so it is never handicapped by a rate mismatch.
    sweep: list[dict[str, float]] = []
    for rate in cfg.mask_rate_sweep:
        masks = make_eval_masks(
            test_data["features"].shape[0], n_sites, rate, cfg.n_eval_repeats, seed + 4231, device,
            cfg.block_missing, cfg.block_len,
        )
        model_eval = evaluate_model(model, test_data, masks, cfg, use_pop_model)
        sweep.append(
            {
                "mask_rate": float(rate),
                "model": model_eval["accuracy"],
                "model_dosage_r2": model_eval["dosage_r2"],
                "majority": majority_baseline(train_data["labels"], test_data["labels"], masks)["accuracy"],
                "explicit_ld": fit_ld_regression(rate).score(
                    test_data["features"], test_data["labels"], masks
                )["accuracy"],
            }
        )

    # ---- LD-structure recovery ----------------------------------------------
    positions = sim["positions"]
    abs_dist = np.abs(positions[:, None] - positions[None, :])
    dist_flat = upper_offdiag(abs_dist)  # magnitude, used as the control variable

    # Correlate per-population (population-specific LD) then average.
    attn_p, bias_p, dist_p, attn_s, bias_s = [], [], [], [], []
    attn_pp, bias_pp = [], []  # distance-controlled (partial) correlations
    r2_mats, attn_mats, bias_mats = [], [], []
    for p in range(len(sim["r2_per_pop"])):
        pop_filter = p if multi_pop_data else None
        if multi_pop_data:
            pop_rows = train_idx[data["pop"].cpu().numpy()[train_idx] == p]
            r2 = compute_true_r2(haplotypes_for(haplotypes, pop_rows))
        else:
            r2 = r2_train
        attn = extract_mean_attention(
            model, train_data, cfg.batch_size, use_pop_model, pop_filter, cfg.use_missing_channel
        )
        bias = extract_mean_bias(
            model, train_data, cfg.batch_size, use_pop_model, pop_filter, cfg.use_missing_channel
        )
        r2_flat = upper_offdiag(r2)
        attn_flat, bias_flat = upper_offdiag(attn), upper_offdiag(bias)
        attn_p.append(pearson(attn_flat, r2_flat))
        attn_s.append(spearman(attn_flat, r2_flat))
        bias_p.append(pearson(bias_flat, r2_flat))
        bias_s.append(spearman(bias_flat, r2_flat))
        # LD captured *beyond* genomic distance:
        attn_pp.append(partial_pearson(attn_flat, r2_flat, dist_flat))
        bias_pp.append(partial_pearson(bias_flat, r2_flat, dist_flat))
        dist_p.append(pearson(-dist_flat, r2_flat))
        r2_mats.append(r2)
        attn_mats.append(attn)
        bias_mats.append(bias)

    return {
        "config": cfg.name,
        "seed": seed,
        "imputation_accuracy": test_eval["accuracy"],
        "dosage_r2": test_eval["dosage_r2"],
        "train_accuracy": fit["train_accuracy"],
        "val_accuracy": fit["val_accuracy"],
        "best_epoch": fit["best_epoch"],
        "baseline_majority_accuracy": majority["accuracy"],
        "baseline_explicit_ld_accuracy": ld_reg_eval["accuracy"],
        "model_minus_explicit_ld": test_eval["accuracy"] - ld_reg_eval["accuracy"],
        "model_minus_majority": test_eval["accuracy"] - majority["accuracy"],
        "explicit_r2_seconds": r2_seconds,
        "mask_rate_sweep": sweep,
        **_maf_stratified_accuracy(test_eval["correct"], test_eval["sites"], maf),
        **_maf_stratified_accuracy(
            ld_reg_eval["correct"], ld_reg_eval["sites"], maf, prefix="baseline_explicit_ld_"
        ),
        **_maf_stratified_accuracy(
            majority["correct"], majority["sites"], maf, prefix="baseline_majority_"
        ),
        "attention_vs_r2_pearson": float(np.mean(attn_p)),
        "attention_vs_r2_spearman": float(np.mean(attn_s)),
        "bias_vs_r2_pearson": float(np.mean(bias_p)),
        "bias_vs_r2_spearman": float(np.mean(bias_s)),
        "attention_vs_r2_partial_pearson": float(np.mean(attn_pp)),
        "bias_vs_r2_partial_pearson": float(np.mean(bias_pp)),
        "distance_baseline_vs_r2_pearson": float(np.mean(dist_p)),
        # matrices for figures (population 0 only, to keep artifacts small)
        "_r2_matrix": r2_mats[0],
        "_attention_matrix": attn_mats[0],
        "_bias_matrix": bias_mats[0],
        "_abs_dist": abs_dist,
    }
