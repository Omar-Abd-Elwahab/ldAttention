"""Tests for the validation library (correlations, residualization, run_config)."""

from __future__ import annotations

import numpy as np
import torch

from ldattention.validation import (
    RunConfig,
    compute_true_r2,
    partial_pearson,
    pearson,
    residualize_by,
    run_config,
    simulate_haplotypes_copying,
    spearman,
)


def test_pearson_and_spearman_basics():
    x = np.arange(50, dtype=np.float64)
    assert pearson(x, 2 * x + 1) > 0.999
    assert pearson(x, -x) < -0.999
    # Spearman captures monotone non-linear relationships.
    assert spearman(x, x**3) > 0.999


def test_residualize_removes_control():
    rng = np.random.default_rng(0)
    control = rng.uniform(size=2000)
    # A pure function of the control should be almost fully removed.
    signal = 3.0 * control
    resid = residualize_by(signal, control, n_bins=16)
    assert np.abs(resid).mean() < 0.1 * np.abs(signal).mean()


def test_partial_pearson_controls_for_confound():
    rng = np.random.default_rng(1)
    control = rng.uniform(size=4000)
    shared = rng.normal(size=4000)
    a = control + shared
    b = 2 * control + shared
    # Raw correlation is inflated by the shared control...
    assert pearson(a, b) > 0.5
    # ...and a real residual association remains after removing it.
    assert partial_pearson(a, b, control) > 0.4


def test_partial_pearson_kills_distance_only_confound():
    rng = np.random.default_rng(2)
    control = rng.uniform(size=8000)
    # A bin-structured confound (as distance is used) is fully removable by the
    # matching-resolution residualization; only independent noise should remain.
    binned_effect = np.floor(control * 16)
    a = binned_effect + rng.normal(size=8000)
    b = binned_effect + rng.normal(size=8000)
    assert pearson(a, b) > 0.3  # raw correlation is inflated by the shared confound
    assert abs(partial_pearson(a, b, control, n_bins=16)) < 0.1  # removed after control


def test_true_r2_is_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    hap, _ = simulate_haplotypes_copying(200, 24, n_founders=4, rho=15.0, mutation_rate=0.01, rng=rng)
    r2 = compute_true_r2(hap)
    assert r2.shape == (24, 24)
    assert np.allclose(r2, r2.T, atol=1e-8)
    assert r2.min() >= 0.0 and r2.max() <= 1.0 + 1e-8


def test_run_config_smoke():
    cfg = RunConfig(
        name="both", n_haplotypes=80, n_sites=16, n_founders=4, rho=15.0,
        hidden_dim=32, num_heads=2, num_layers=1, epochs=3, batch_size=16,
    )
    res = run_config(cfg, seed=0, device=torch.device("cpu"))
    for key in (
        "imputation_accuracy",
        "attention_vs_r2_pearson",
        "bias_vs_r2_pearson",
        "attention_vs_r2_partial_pearson",
        "bias_vs_r2_partial_pearson",
    ):
        assert key in res and np.isfinite(res[key])
    assert res["_r2_matrix"].shape == (16, 16)
    assert res["_bias_matrix"].shape == (16, 16)
