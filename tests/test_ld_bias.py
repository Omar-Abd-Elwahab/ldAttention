"""Correctness tests for the LD-aware attention bias primitive.

These encode the invariants that make the mechanism genuinely "LD-aware":
symmetry, distance decay, and that every learned term actually affects the
softmax (guarding against the class of silent no-op bias bugs).
"""

from __future__ import annotations

import pytest
import torch

from ldattention.integrations import LDAwareMultiheadAttention, ld_bias_to_mha_mask
from ldattention.models.ld_attention import LDAwareSelfAttention
from ldattention.models.ld_bias import LDAttentionBias

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _positions(batch: int, seq_len: int) -> torch.Tensor:
    return torch.linspace(0, 1, seq_len).view(1, seq_len, 1).repeat(batch, 1, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_bias_shape_and_finiteness(device):
    b, l, h, c = 2, 16, 4, 32
    module = LDAttentionBias(hidden_dim=c, num_heads=h).to(device)
    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    bias = module(positions, token_embeddings=x)
    assert bias.shape == (b, h, l, l)
    assert torch.isfinite(bias).all()


@pytest.mark.parametrize("device", DEVICES)
def test_distance_and_genotype_terms_are_symmetric(device):
    b, l, h, c = 2, 12, 4, 32
    # Distance-only term must be exactly symmetric.
    dist_only = LDAttentionBias(hidden_dim=c, num_heads=h, use_genotype_bias=False).to(device)
    positions = _positions(b, l).to(device)
    bias = dist_only(positions)
    assert torch.allclose(bias, bias.transpose(-1, -2), atol=1e-6)

    # Genotype term is a symmetric bilinear form; full bias stays symmetric.
    full = LDAttentionBias(hidden_dim=c, num_heads=h).to(device)
    x = torch.randn(b, l, c, device=device)
    bias_full = full(positions, token_embeddings=x)
    assert torch.allclose(bias_full, bias_full.transpose(-1, -2), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_distance_bias_can_learn_decay(device):
    """After fitting toward a decaying target, near pairs should outscore far pairs."""
    b, l, h, c = 1, 32, 2, 16
    module = LDAttentionBias(hidden_dim=c, num_heads=h, use_genotype_bias=False).to(device)
    positions = _positions(b, l).to(device)

    # Target: bias decays linearly with distance (proxy for LD decay).
    dist = (positions.squeeze(-1)[:, :, None] - positions.squeeze(-1)[:, None, :]).abs()
    target = (-dist)[:, None].expand(b, h, l, l)

    opt = torch.optim.Adam(module.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.mean((module(positions) - target) ** 2)
        loss.backward()
        opt.step()

    learned = module(positions)
    near = learned[0, 0, 0, 1]
    far = learned[0, 0, 0, l - 1]
    assert near > far


@pytest.mark.parametrize("device", DEVICES)
def test_genotype_term_is_not_a_no_op(device):
    """The genotype term must change attention (guards the softmax no-op bug)."""
    b, l, h, c = 2, 16, 4, 32
    dist_only = LDAttentionBias(hidden_dim=c, num_heads=h, use_genotype_bias=False).to(device)
    full = LDAttentionBias(hidden_dim=c, num_heads=h, use_genotype_bias=True).to(device)
    # Copy the distance parameters so the *only* difference is the genotype term.
    full.distance_bias.load_state_dict(dist_only.distance_bias.state_dict())

    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    with torch.no_grad():
        b0 = dist_only(positions, token_embeddings=x)
        b1 = full(positions, token_embeddings=x)
    # Distinct logits...
    assert not torch.allclose(b0, b1)
    # ...and distinct softmax rows (i.e. not a per-row constant that cancels).
    p0 = torch.softmax(b0, dim=-1)
    p1 = torch.softmax(b1, dim=-1)
    assert not torch.allclose(p0, p1, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_flow_to_every_term(device):
    b, l, h, c = 2, 16, 4, 32
    module = LDAttentionBias(hidden_dim=c, num_heads=h, num_populations=3).to(device)
    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    pop = torch.tensor([0, 1], device=device)
    bias = module(positions, token_embeddings=x, population_id=pop)
    bias.sum().backward()
    assert module.distance_bias.weight.grad is not None
    assert module.distance_bias.weight.grad.abs().sum() > 0
    assert module.geno_proj.weight.grad.abs().sum() > 0
    assert module.pop_film.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_key_padding_mask_zeroes_attention(device):
    b, l, h, c = 2, 10, 2, 16
    module = LDAttentionBias(hidden_dim=c, num_heads=h).to(device)
    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    keep = torch.ones(b, l, dtype=torch.bool, device=device)
    keep[:, l // 2 :] = False  # mask the second half
    bias = module(positions, token_embeddings=x, key_padding_mask=keep)
    weights = torch.softmax(bias, dim=-1)
    assert weights[..., l // 2 :].max() < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_self_attention_sdpa_and_manual_agree(device):
    torch.manual_seed(0)
    b, l, c, h = 2, 16, 32, 4
    attn = LDAwareSelfAttention(hidden_dim=c, num_heads=h, dropout=0.0).to(device).eval()
    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    out_sdpa, w_sdpa = attn(x, positions, need_weights=False)
    out_manual, w_manual = attn(x, positions, need_weights=True)
    assert w_sdpa is None
    assert w_manual is not None and w_manual.shape == (b, h, l, l)
    assert torch.allclose(out_sdpa, out_manual, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_mha_adapter_runs(device):
    b, l, c, h = 2, 16, 32, 4
    layer = LDAwareMultiheadAttention(embed_dim=c, num_heads=h, dropout=0.0).to(device)
    positions = _positions(b, l).to(device)
    x = torch.randn(b, l, c, device=device)
    keep = torch.ones(b, l, dtype=torch.bool, device=device)
    out, weights = layer(x, positions, key_padding_mask=keep, need_weights=True)
    assert out.shape == (b, l, c)
    assert weights.shape == (b, h, l, l)


def test_ld_bias_to_mha_mask_shape():
    bias = torch.randn(2, 4, 8, 8)
    mask = ld_bias_to_mha_mask(bias)
    assert mask.shape == (8, 8, 8)
