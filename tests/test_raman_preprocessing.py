import numpy as np
import pytest

from pgsg2.ingestion.base import SpectralDataset
from pgsg2.preprocessing.raman import (
    RamanPreprocessor,
    als_baseline,
    remove_cosmic_rays,
    snv,
)


def _gaussian_peak(x, center, height, width):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def _make_synthetic_spectrum(p=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, p - 1, p)
    peaks = (
        _gaussian_peak(x, 80, 10.0, 4.0)
        + _gaussian_peak(x, 180, 6.0, 3.0)
        + _gaussian_peak(x, 240, 8.0, 5.0)
    )
    baseline = 0.01 * (x - p / 2) ** 2 / p
    noise = rng.normal(0, 0.05, size=p)
    return x, peaks, baseline, noise


def test_als_baseline_follows_smooth_drift_not_peaks():
    x, peaks, baseline, noise = _make_synthetic_spectrum()
    y = peaks + baseline + noise
    estimated = als_baseline(y, lam=1e5, p=0.01, niter=10)
    residual_vs_true_baseline = np.abs(estimated - baseline)
    assert residual_vs_true_baseline.mean() < 1.0
    peak_idx = np.argmax(peaks)
    assert estimated[peak_idx] < peaks[peak_idx]


def test_als_baseline_handles_short_spectrum():
    y = np.array([1.0, 2.0])
    baseline = als_baseline(y)
    assert baseline.shape == y.shape


def test_remove_cosmic_rays_suppresses_single_spike():
    x, peaks, baseline, noise = _make_synthetic_spectrum()
    y = peaks + baseline + noise
    y_spiked = y.copy()
    spike_idx = 150
    y_spiked[spike_idx] += 50.0
    y_clean = remove_cosmic_rays(y_spiked, threshold=8.0, window=5)
    assert y_clean[spike_idx] < y_spiked[spike_idx] / 2
    mask = np.ones_like(y, dtype=bool)
    mask[spike_idx - 1 : spike_idx + 2] = False
    np.testing.assert_allclose(y_clean[mask], y_spiked[mask], atol=1e-9)


def test_remove_cosmic_rays_leaves_clean_spectrum_untouched():
    x, peaks, baseline, noise = _make_synthetic_spectrum(seed=1)
    y = peaks + baseline + noise
    y_clean = remove_cosmic_rays(y, threshold=8.0, window=5)
    n_diff = np.sum(~np.isclose(y_clean, y))
    assert n_diff <= max(1, int(0.01 * len(y)))


def test_snv_produces_zero_mean_unit_std():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_snv = snv(y)
    assert abs(y_snv.mean()) < 1e-9
    assert abs(y_snv.std() - 1.0) < 1e-9


def test_snv_handles_constant_spectrum_without_nan():
    y = np.full(10, 3.0)
    y_snv = snv(y)
    assert not np.isnan(y_snv).any()


def _make_dataset(n=6, p=300, with_spike=False, seed=0):
    rows = []
    for i in range(n):
        x, peaks, baseline, noise = _make_synthetic_spectrum(p=p, seed=seed + i)
        y = peaks + baseline + noise
        if with_spike:
            y[100] += 40.0
        rows.append(y)
    X = np.vstack(rows)
    y_target = np.linspace(1.0, 2.0, n)
    return SpectralDataset(
        X=X, y=y_target, wavelengths=x, domain="raman",
        wavelength_unit="cm-1", target_name="glucose",
        metadata={"source": "synthetic"},
    )


def test_preprocessor_returns_valid_dataset_with_same_shape():
    ds = _make_dataset(with_spike=True)
    pre = RamanPreprocessor()
    ds_out = pre.fit_transform(ds)
    assert isinstance(ds_out, SpectralDataset)
    assert ds_out.X.shape == ds.X.shape
    np.testing.assert_array_equal(ds_out.y, ds.y)
    np.testing.assert_array_equal(ds_out.wavelengths, ds.wavelengths)
    assert ds_out.domain == "raman"
    assert not np.isnan(ds_out.X).any()


def test_preprocessor_output_is_approximately_zero_mean_per_spectrum():
    ds = _make_dataset()
    pre = RamanPreprocessor()
    ds_out = pre.fit_transform(ds)
    means = ds_out.X.mean(axis=1)
    np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-6)


def test_preprocessor_rejects_non_raman_domain():
    x = np.linspace(900, 1700, 50)
    ds_nir = SpectralDataset(
        X=np.ones((3, 50)), y=np.array([1.0, 2.0, 3.0]), wavelengths=x,
        domain="nir", wavelength_unit="nm", target_name="dmc",
    )
    pre = RamanPreprocessor()
    with pytest.raises(ValueError):
        pre.transform(ds_nir)
