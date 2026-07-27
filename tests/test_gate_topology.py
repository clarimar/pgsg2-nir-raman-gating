"""
Testes de contrato dos descritores de topologia do gate.

Usam casos conhecidos analiticamente (gate uniforme, one-hot, etc.) para
verificar que as fórmulas batem com os valores esperados, não só que
"rodam sem erro".
"""

import numpy as np
import pytest

from pgsg2.interpretability.topology import (
    gate_entropy,
    gate_smoothness,
    gate_sparsity_hoyer,
    gate_topology,
)


# ---------- gate_entropy ----------

def test_entropy_uniform_gate_is_log_p():
    p = 100
    g = np.full(p, 1.0 / p)
    H = gate_entropy(g)
    np.testing.assert_allclose(H, np.log(p), rtol=1e-6)


def test_entropy_near_one_hot_is_near_zero():
    p = 50
    g = np.full(p, 1e-6)
    g[0] = 1.0 - 1e-6 * (p - 1)
    H = gate_entropy(g)
    assert H < 0.01


def test_entropy_uniform_is_maximum_among_valid_gates():
    p = 30
    rng = np.random.default_rng(0)
    g_uniform = np.full(p, 1.0 / p)
    H_uniform = gate_entropy(g_uniform)

    for _ in range(5):
        theta = rng.normal(size=p)
        g_random = np.exp(theta) / np.exp(theta).sum()
        H_random = gate_entropy(g_random)
        assert H_random <= H_uniform + 1e-9


def test_entropy_renormalizes_with_warning_if_not_summing_to_one():
    g = np.array([0.5, 0.5, 0.5, 0.5])  # soma 2, não é um gate softmax válido
    with pytest.warns(UserWarning):
        H = gate_entropy(g)
    # equivalente a um gate uniforme de 4 elementos após renormalização
    np.testing.assert_allclose(H, np.log(4), rtol=1e-6)


def test_entropy_raises_if_renormalize_disabled_and_sum_off():
    g = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        gate_entropy(g, renormalize_if_needed=False)


# ---------- gate_smoothness ----------

def test_smoothness_constant_gate_is_zero():
    g = np.full(20, 1.0 / 20)
    assert gate_smoothness(g) == pytest.approx(0.0)


def test_smoothness_alternating_gate_is_high():
    p = 20
    g = np.array([0.1 if i % 2 == 0 else 0.9 for i in range(p)])
    g = g / g.sum()
    tv_alternating = gate_smoothness(g)

    g_smooth = np.linspace(0.01, 0.99, p)
    g_smooth = g_smooth / g_smooth.sum()
    tv_smooth = gate_smoothness(g_smooth)

    assert tv_alternating > tv_smooth


def test_smoothness_requires_at_least_two_bands():
    with pytest.raises(ValueError):
        gate_smoothness(np.array([1.0]))


# ---------- gate_sparsity_hoyer ----------

def test_sparsity_uniform_gate_is_zero():
    p = 64
    g = np.full(p, 1.0 / p)
    S = gate_sparsity_hoyer(g)
    assert S == pytest.approx(0.0, abs=1e-9)


def test_sparsity_one_hot_gate_is_one():
    p = 64
    g = np.zeros(p)
    g[0] = 1.0
    S = gate_sparsity_hoyer(g)
    assert S == pytest.approx(1.0, abs=1e-9)


def test_sparsity_intermediate_case_between_zero_and_one():
    p = 64
    rng = np.random.default_rng(1)
    theta = rng.normal(size=p)
    g = np.exp(theta) / np.exp(theta).sum()
    S = gate_sparsity_hoyer(g)
    assert 0.0 < S < 1.0


def test_sparsity_monotonic_as_gate_concentrates():
    """Concentrar peso progressivamente numa única banda deve aumentar S."""
    p = 32
    base = np.full(p, 1.0 / p)

    def concentrate(alpha):
        g = base.copy()
        g[0] += alpha
        g = np.clip(g, 0, None)
        return g / g.sum()

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8]
    sparsities = [gate_sparsity_hoyer(concentrate(a)) for a in alphas]
    assert all(s2 >= s1 - 1e-9 for s1, s2 in zip(sparsities, sparsities[1:]))


def test_sparsity_handles_all_zero_gate_gracefully():
    g = np.zeros(10)
    assert gate_sparsity_hoyer(g) == 0.0


# ---------- gate_topology (conveniência) ----------

def test_gate_topology_returns_all_three_keys():
    g = np.full(40, 1.0 / 40)
    topo = gate_topology(g)
    assert set(topo.keys()) == {"entropy", "smoothness", "sparsity_hoyer"}
    assert topo["entropy"] == pytest.approx(np.log(40), rel=1e-6)
    assert topo["smoothness"] == pytest.approx(0.0)
    assert topo["sparsity_hoyer"] == pytest.approx(0.0, abs=1e-9)


# ---------- validações comuns ----------

def test_rejects_negative_values():
    g = np.array([0.5, -0.1, 0.6])
    with pytest.raises(ValueError):
        gate_entropy(g)
    with pytest.raises(ValueError):
        gate_smoothness(g)
    with pytest.raises(ValueError):
        gate_sparsity_hoyer(g)


def test_rejects_nan():
    g = np.array([0.5, np.nan, 0.5])
    with pytest.raises(ValueError):
        gate_entropy(g)
