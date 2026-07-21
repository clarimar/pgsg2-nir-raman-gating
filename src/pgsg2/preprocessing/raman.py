"""
Operador T (pré-processamento) do ramo Raman de pgsg_2: remoção de
cosmic rays, correção de linha de base (ALS) e normalização (SNV).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pgsg2.ingestion.base import SpectralDataset


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Linha de base por Asymmetric Least Squares (Eilers & Boelens, 2005)."""
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def remove_cosmic_rays(
    y: np.ndarray,
    threshold: float = 8.0,
    window: int = 5,
    max_spike_width: int = 3,
) -> np.ndarray:
    """Remove cosmic rays via z-score modificado (Whitaker & Hayes, 2018),
    restrito a agrupamentos contíguos estreitos (<= max_spike_width) para
    não confundir borda de pico real com spike."""
    if len(y) < 3:
        return y.copy()

    dy = np.diff(y)
    median_dy = np.median(dy)
    mad = np.median(np.abs(dy - median_dy))
    scale = mad if mad > 1e-12 else 1e-12
    modified_z = 0.6745 * (dy - median_dy) / scale

    flagged = np.abs(modified_z) > threshold
    flagged_idx = np.where(flagged)[0] + 1

    spike_idx: set[int] = set()
    if flagged_idx.size:
        groups = np.split(flagged_idx, np.where(np.diff(flagged_idx) > 1)[0] + 1)
        for group in groups:
            if len(group) <= max_spike_width:
                spike_idx.update(group.tolist())

    y_clean = y.copy()
    for idx in sorted(spike_idx):
        lo, hi = max(0, idx - window), min(len(y), idx + window + 1)
        neighbors = [i for i in range(lo, hi) if i != idx and i not in spike_idx]
        if neighbors:
            y_clean[idx] = np.mean(y[neighbors])
    return y_clean


def snv(y: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: (y - média) / desvio-padrão, por espectro."""
    std = y.std()
    return (y - y.mean()) / (std if std > 1e-12 else 1e-12)


@dataclass
class RamanPreprocessor:
    als_lam: float = 1e5
    als_p: float = 0.01
    als_niter: int = 10
    cosmic_ray_threshold: float = 8.0
    cosmic_ray_window: int = 5

    def fit(self, dataset: SpectralDataset) -> "RamanPreprocessor":
        return self

    def transform(self, dataset: SpectralDataset) -> SpectralDataset:
        if dataset.domain != "raman":
            raise ValueError(
                f"RamanPreprocessor só se aplica a domain='raman'; recebido {dataset.domain!r}"
            )

        X_out = np.empty_like(dataset.X)
        for i in range(dataset.X.shape[0]):
            spectrum = dataset.X[i]
            despiked = remove_cosmic_rays(
                spectrum, threshold=self.cosmic_ray_threshold, window=self.cosmic_ray_window
            )
            baseline = als_baseline(despiked, lam=self.als_lam, p=self.als_p, niter=self.als_niter)
            corrected = despiked - baseline
            X_out[i] = snv(corrected)

        return replace(dataset, X=X_out)

    def fit_transform(self, dataset: SpectralDataset) -> SpectralDataset:
        return self.fit(dataset).transform(dataset)
