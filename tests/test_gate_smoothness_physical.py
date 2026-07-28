import numpy as np
import pytest

from pgsg2.interpretability.topology import gate_smoothness_physical


def test_matches_raw_tv_when_spacing_is_uniform_unit():
    """Com espaçamento unitário, deve coincidir com a TV bruta."""
    from pgsg2.interpretability.topology import gate_smoothness
    g = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
    wl = np.arange(len(g), dtype=float)  # espaçamento 1.0
    tv_raw = gate_smoothness(g)
    tv_phys = gate_smoothness_physical(g, wl)
    assert tv_phys == pytest.approx(tv_raw)


def test_denser_bands_reduce_physical_smoothness_for_same_gate_shape():
    """O mesmo formato de gate, amostrado em bandas mais densas (mesma
    faixa física, mais pontos), deve ter menor variação POR UNIDADE
    FÍSICA quando a densidade de bandas é maior -- controla o
    confundidor de resolução."""
    # gate suave subindo e descendo ao longo de uma faixa de 100 unidades físicas
    wl_sparse = np.linspace(0, 100, 20)
    g_sparse = np.sin(wl_sparse / 100 * np.pi)  # mesma forma física

    wl_dense = np.linspace(0, 100, 200)
    g_dense = np.sin(wl_dense / 100 * np.pi)

    tv_phys_sparse = gate_smoothness_physical(g_sparse, wl_sparse)
    tv_phys_dense = gate_smoothness_physical(g_dense, wl_dense)
    # normalizada pela física, as duas devem ficar próximas (mesma forma,
    # densidade de amostragem não deveria inflar/deflacionar a métrica)
    assert tv_phys_sparse == pytest.approx(tv_phys_dense, rel=0.15)


def test_rejects_mismatched_shapes():
    g = np.array([0.1, 0.2, 0.3])
    wl = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        gate_smoothness_physical(g, wl)


def test_rejects_duplicate_wavelengths():
    g = np.array([0.1, 0.2, 0.3])
    wl = np.array([1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        gate_smoothness_physical(g, wl)
