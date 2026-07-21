from dataclasses import dataclass, field

import numpy as np
import pytest

from pgsg2.ingestion.base import SpectralDataset
from pgsg2.ingestion.raman_bioprocess import (
    RamanBioprocessSubstratesLoader,
    RamanTargetNotFoundError,
    TARGET_NAME,
    _find_target_index,
)


@dataclass
class _FakeRamanDataset:
    spectra: np.ndarray
    targets: np.ndarray
    raman_shifts: np.ndarray
    target_names: list[str] = field(default_factory=list)


def _make_fake_loader(fake_dataset, dataset_id_expected="bioprocess_substrates"):
    def _loader_fn(dataset_id, cache_dir=None):
        assert dataset_id == dataset_id_expected
        return fake_dataset

    return _loader_fn


def test_load_returns_valid_spectral_dataset():
    n, p = 50, 300
    rng = np.random.default_rng(0)
    fake = _FakeRamanDataset(
        spectra=rng.random((n, p)),
        targets=np.column_stack([rng.random(n) for _ in range(8)]),
        raman_shifts=np.linspace(200, 3200, p),
        target_names=[
            "glycerol", "glucose", "acetate", "ammonium",
            "phosphate", "magnesium_sulfate", "lactate", "antifoam",
        ],
    )
    loader = RamanBioprocessSubstratesLoader(loader_fn=_make_fake_loader(fake))
    ds = loader.load()

    assert isinstance(ds, SpectralDataset)
    assert ds.domain == "raman"
    assert ds.wavelength_unit == "cm-1"
    assert ds.target_name == TARGET_NAME
    assert ds.n_samples == n
    assert ds.n_bands == p
    np.testing.assert_allclose(ds.y, fake.targets[:, 1])
    assert ds.metadata["source_dataset_id"] == "bioprocess_substrates"
    assert ds.metadata["n_samples_dropped_missing_target"] == 0


def test_load_drops_samples_with_missing_target():
    n, p = 10, 20
    targets = np.zeros((n, 2))
    targets[:, 0] = 1.0
    targets[3, 0] = np.nan
    targets[7, 0] = np.nan
    fake = _FakeRamanDataset(
        spectra=np.ones((n, p)),
        targets=targets,
        raman_shifts=np.linspace(200, 3200, p),
        target_names=["glucose", "acetate"],
    )
    loader = RamanBioprocessSubstratesLoader(loader_fn=_make_fake_loader(fake))
    ds = loader.load()

    assert ds.n_samples == n - 2
    assert ds.metadata["n_samples_dropped_missing_target"] == 2
    assert not np.isnan(ds.y).any()


def test_load_raises_when_target_absent():
    n, p = 5, 10
    fake = _FakeRamanDataset(
        spectra=np.ones((n, p)),
        targets=np.ones((n, 2)),
        raman_shifts=np.linspace(200, 3200, p),
        target_names=["acetate", "lactate"],
    )
    loader = RamanBioprocessSubstratesLoader(loader_fn=_make_fake_loader(fake))
    with pytest.raises(RamanTargetNotFoundError):
        loader.load()


def test_load_raises_on_ambiguous_target_name():
    n, p = 5, 10
    fake = _FakeRamanDataset(
        spectra=np.ones((n, p)),
        targets=np.ones((n, 2)),
        raman_shifts=np.linspace(200, 3200, p),
        target_names=["glucose_feed", "glucose_residual"],
    )
    loader = RamanBioprocessSubstratesLoader(loader_fn=_make_fake_loader(fake))
    with pytest.raises(RamanTargetNotFoundError):
        loader.load()


def test_load_raises_when_loader_returns_none():
    loader = RamanBioprocessSubstratesLoader(loader_fn=lambda dsid, cache_dir=None: None)
    with pytest.raises(RuntimeError):
        loader.load()


def test_find_target_index_case_insensitive():
    assert _find_target_index(["Glucose", "Acetate"], "glucose") == 0


@pytest.mark.integration
def test_load_real_dataset_via_raman_data():
    pytest.importorskip("raman_data")
    loader = RamanBioprocessSubstratesLoader()
    ds = loader.load()
    assert ds.domain == "raman"
    assert ds.n_samples > 0
    assert ds.n_bands == ds.wavelengths.shape[0]
