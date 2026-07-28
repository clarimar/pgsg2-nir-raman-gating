import numpy as np
import pytest

from pgsg2.priors.nir_gaussian import NIRLiteratureGaussianPrior


def test_signature_has_no_data_dependent_arguments():
    import inspect
    sig = inspect.signature(NIRLiteratureGaussianPrior.compute)
    assert list(sig.parameters.keys()) == ["self", "wavelengths"]


def test_values_within_valid_range():
    wl = np.linspace(285, 1200, 306)
    prior = NIRLiteratureGaussianPrior().compute(wl)
    assert prior.shape == wl.shape
    assert (prior > 0).all()
    assert (prior <= 1.0).all()


def test_peaks_near_literature_bands():
    wl = np.linspace(285, 1200, 1000)
    prior = NIRLiteratureGaussianPrior().compute(wl)

    idx_950 = np.argmin(np.abs(wl - 950))
    idx_far = np.argmin(np.abs(wl - 300))  # longe de qualquer banda
    assert prior[idx_950] > prior[idx_far]


def test_950nm_band_is_strongest_matching_original_weights():
    """make_literature_prior original dá peso 1.0 a 900-1000nm (o maior);
    a versão gaussiana deve preservar essa hierarquia relativa."""
    wl = np.linspace(285, 1200, 1000)
    prior = NIRLiteratureGaussianPrior().compute(wl)

    idx_690 = np.argmin(np.abs(wl - 690))
    idx_950 = np.argmin(np.abs(wl - 950))
    idx_1150 = np.argmin(np.abs(wl - 1150))

    assert prior[idx_950] >= prior[idx_690]
    assert prior[idx_950] >= prior[idx_1150]


def test_no_abrupt_discontinuities_unlike_step_function():
    """Ao contrário do degrau original (saltos de ~0.4-0.9 entre bandas
    vizinhas nas fronteiras de região), a versão gaussiana deve ter
    transições bem mais suaves."""
    wl = np.linspace(285, 1200, 306)  # mesma resolucao real do Mango
    prior = NIRLiteratureGaussianPrior().compute(wl)
    max_jump = np.max(np.abs(np.diff(prior)))
    assert max_jump < 0.15  # bem menor que o salto do degrau original (~0.4-0.9)
