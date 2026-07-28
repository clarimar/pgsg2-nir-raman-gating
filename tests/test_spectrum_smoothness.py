import numpy as np
import pytest

from pgsg2.interpretability.topology import spectrum_smoothness


def test_constant_spectra_have_zero_smoothness():
    X = np.ones((5, 50))
    assert spectrum_smoothness(X) == pytest.approx(0.0)


def test_noisier_spectrum_has_higher_smoothness_value():
    rng = np.random.default_rng(0)
    p = 200
    smooth = np.tile(np.sin(np.linspace(0, 2 * np.pi, p)), (10, 1))
    noisy = smooth + rng.normal(0, 1.0, size=(10, p))

    tv_smooth = spectrum_smoothness(smooth)
    tv_noisy = spectrum_smoothness(noisy)
    assert tv_noisy > tv_smooth


def test_rejects_1d_input():
    with pytest.raises(ValueError):
        spectrum_smoothness(np.ones(50))


def test_rejects_too_few_bands():
    with pytest.raises(ValueError):
        spectrum_smoothness(np.ones((5, 1)))


def test_averages_over_samples_correctly():
    # duas amostras com TV conhecida: [0,1,0,1] -> diffs=[1,1,1], mean=1
    #                                  [0,0,0,0] -> diffs=[0,0,0], mean=0
    X = np.array([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=float)
    result = spectrum_smoothness(X)
    assert result == pytest.approx((1.0 + 0.0) / 2)
