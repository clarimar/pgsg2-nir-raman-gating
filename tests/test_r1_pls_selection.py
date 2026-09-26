"""Unit tests for the PLS component-selection helpers of r1_h1_calibrated_pls.py."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "r1", Path(__file__).resolve().parents[1] / "scripts" / "r1_h1_calibrated_pls.py")
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)


def _latent_data(n=300, p=80, k=3, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.normal(size=(n, k))
    X = T @ rng.normal(size=(k, p)) + noise * rng.normal(size=(n, p))
    y = T @ np.ones(k) + noise * rng.normal(size=n)
    return X, y


def test_select_h_rule():
    hs = np.arange(1, 6)
    mean = np.array([5.0, 2.0, 1.05, 1.0, 1.02])
    se = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    assert r1.select_h(hs, mean, se) == (4, 3)


def test_1se_never_exceeds_cvmin():
    X, y = _latent_data()
    hs, mean, se = r1.pls_cv_curve(X, y, 15, 5, seed=1)
    h_min, h_1se = r1.select_h(hs, mean, se)
    assert h_1se <= h_min


def test_recovers_true_rank():
    X, y = _latent_data(k=3, noise=0.01)
    hs, mean, se = r1.pls_cv_curve(X, y, 15, 5, seed=2)
    _, h_1se = r1.select_h(hs, mean, se)
    assert h_1se == 3


def test_cv_uses_training_only_deterministic():
    X, y = _latent_data(seed=3)
    a = r1.pls_cv_curve(X, y, 8, 5, seed=4)
    b = r1.pls_cv_curve(X, y, 8, 5, seed=4)
    assert np.array_equal(a[1], b[1])


import pytest


@pytest.mark.parametrize("scale", [False, True])
def test_truncation_matches_direct_fit(scale):
    from sklearn.cross_decomposition import PLSRegression
    X, y = _latent_data(n=200, p=60, k=6, noise=0.1, seed=5)
    X = X * np.linspace(0.1, 10, X.shape[1])  # unequal column variances
    big = PLSRegression(n_components=20, scale=scale).fit(X, y)
    for h in (1, 3, 7, 20):
        direct = PLSRegression(n_components=h, scale=scale).fit(X, y).predict(X).ravel()
        assert np.allclose(r1._truncated_predict(big, X, h), direct, atol=1e-8)


def test_group_split_keeps_groups_intact():
    groups = np.repeat(np.arange(100), 2)
    tr, te = r1.group_split(groups, 0.2, seed=3)
    assert set(groups[tr]).isdisjoint(groups[te])
    assert len(set(groups[te])) == 20 and len(tr) + len(te) == 200


def test_group_folds_disjoint_and_complete():
    groups = np.r_[np.repeat(np.arange(50), 2), np.repeat(np.arange(50, 55), 6)]
    folds = r1.group_folds(groups, 5, seed=1)
    seen = np.concatenate([va for _, va in folds])
    assert np.array_equal(np.sort(seen), np.arange(len(groups)))
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(groups[va])


def test_grouped_cv_runs():
    X, y = _latent_data(n=200)
    groups = np.repeat(np.arange(100), 2)
    hs, mean, se = r1.pls_cv_curve(X, y, 10, 5, seed=0, groups=groups)
    assert mean.shape == (10,) and np.all(np.isfinite(mean))
