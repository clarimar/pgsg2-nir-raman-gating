"""
R1 (CHEMOLAB-D-26-01046) -- H1 re-evaluated against a calibrated PLS baseline.

Why this script exists
----------------------
The submitted H1 compares a single PGSGv2Model run (split seed 42, model
seed 42) against PLS with a fixed number of latent variables (H=10),
inherited from the Tecator setting and never re-selected for Mango or
Raman. The H4 runs on the same split show PGSGv2 NIR R2 = 0.7933 +/- 0.0080
over 10 seeds, so the H1 value 0.8128 sits ~2.4 SD above the seed mean.

Design (fixed before running)
-----------------------------
* Branches: NIR (Mango DMC v3, Season 4) and Raman (bioprocess_substrates,
  glucose), identical loading/preprocessing to extract_nir_gate_v2.py and
  run_h1_raman_v2.py.
* Repetitions r = 0..9: random 80/20 split with random_state=r (same
  splitter as the paper, no stratification), PGSGv2Model seed = r.
  Every model is fitted on the same training partition of repetition r
  -> paired comparison.
* Reference repetition "paper": split seed 42, model seed 42. Must
  reproduce the submitted numbers (NIR 0.5684/0.8128, Raman 0.9044/0.9290);
  if not, stop and investigate before interpreting anything.
* Models:
    PLS-10     pgsg_1 PLSModel(n_components=10)          (as submitted)
    PLS-cvmin  sklearn PLS, H = argmin 5-fold CV MSE on training only
    PLS-1se    sklearn PLS, smallest H with CV MSE <= min + 1 SE
    PGSGv2     PGSGv2Model with the literature prior (as submitted)
  H grid 1..H_MAX, folds shuffled with random_state=r. Test data never
  touch model selection.
* Consistency check: sklearn PLS with H=10 must match PLSModel(10) on the
  paper split (|dR2| < 1e-6). Otherwise the cvmin/1se rows are not
  comparable with the PLS-10 row, and the script aborts unless
  --skip-consistency-check is given.
* Decision rule for H1 in a modality: mean paired
  Delta = R2(PGSGv2) - R2(PLS-1se) > 0 AND two-sided Wilcoxon
  signed-rank p < 0.05 over the 10 repetitions. PLS-cvmin is reported
  alongside, not used for the decision.

Grouped splits (added after the replicate diagnostic, r1_check_replicates.py)
-----------------------------------------------------------------------------
Both datasets contain replicate spectra of the same physical sample: Mango
Season 4 has 725 fruits for 1,448 spectra (consecutive rows with identical
metadata and DM), and bioprocess_substrates has 3,191 distinct mixtures
(identical target vectors) for 6,472 spectra. By default the ten evaluation
repetitions therefore split whole samples: test partition by group, and the
PLS cross-validation folds by group within the training partition. The
reference repetition "paper" keeps the original random split so that it can
reproduce the submitted numbers. --ungrouped restores random splits for all
repetitions. Note: PGSGv2Model draws its early-stopping validation subset
internally (pgsg_v2.py) and this script does not change it.

Usage
-----
    python scripts/r1_h1_calibrated_pls.py \\
        --pgsg1-root /home/clarimar/Dropbox/pgsg/pgsg_1 \\
        --csv-path /home/clarimar/Dropbox/pgsg/pgsg_1/data/mango_dmc_v3/MangoDMC_NIR_Data_v3.csv
    # smoke test: --quick (2 repetitions, 30 epochs, H_MAX=15)

Outputs (in --out-dir, default results_r1/)
    r1_h1_runs.csv       one row per (branch, repetition, model)
    r1_h1_selected_H.csv H chosen by cvmin / 1se per (branch, repetition)
    r1_h1_summary.json   means, SDs, paired deltas, Wilcoxon p, decision
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

TEST_FRACTION = 0.2
SEASON = 4
N_REPS = 10
H_MAX = 30
N_FOLDS = 5
# pgsg_1 PLSModel uses PLSRegression(scale=True); the calibrated baselines use the same
# setting so that PLS-10, PLS-cvmin and PLS-1se differ only in H.
PLS_SCALE = True


# --------------------------------------------------------------------------
# PLS component selection (standalone, unit-tested)
# --------------------------------------------------------------------------
def _truncated_predict(model, X, h):
    """Predict with the first h components of a fitted sklearn PLSRegression.

    NIPALS extracts components sequentially and P^T W is upper triangular, so the
    h-component model equals the truncation of a larger model. Verified against
    direct fits in tests/test_r1_pls_selection.py.
    """
    W = model.x_weights_[:, :h]
    P = model.x_loadings_[:, :h]
    Q = model.y_loadings_[:, :h]
    R = W @ np.linalg.pinv(P.T @ W)
    Xc = (X - model._x_mean) / model._x_std
    return (Xc @ R @ Q.T).ravel() * model._y_std.ravel()[0] + model._y_mean.ravel()[0]


def group_split(groups, test_frac, seed):
    """Random split of whole groups into (train_idx, test_idx)."""
    groups = np.asarray(groups)
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    test_groups = u[: int(round(test_frac * len(u)))]
    is_test = np.isin(groups, test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def group_folds(groups, n_folds, seed):
    """K folds of whole groups, groups shuffled then assigned round-robin."""
    groups = np.asarray(groups)
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    fold_of = {g: i % n_folds for i, g in enumerate(u)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(n_folds)]


def pls_cv_curve(X, y, h_max, n_folds, seed, groups=None):
    """Return (hs, mean_mse, se_mse) of K-fold CV over H = 1..h_max.

    One PLS model with h_max components is fitted per fold; smaller H are
    obtained by truncation (exact for NIPALS), which makes large grids cheap.
    """
    h_max = int(min(h_max, X.shape[1], X.shape[0] - X.shape[0] // n_folds - 1))
    hs = np.arange(1, h_max + 1)
    if groups is None:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    else:
        folds = group_folds(groups, n_folds, seed)
    fold_mse = np.zeros((n_folds, h_max))
    for k, (tr, va) in enumerate(folds):
        m = PLSRegression(n_components=h_max, scale=PLS_SCALE).fit(X[tr], y[tr])
        for j, h in enumerate(hs):
            fold_mse[k, j] = np.mean((y[va] - _truncated_predict(m, X[va], int(h))) ** 2)
    mean = fold_mse.mean(axis=0)
    se = fold_mse.std(axis=0, ddof=1) / np.sqrt(n_folds)
    return hs, mean, se


def select_h(hs, mean, se):
    """Return (h_cvmin, h_1se)."""
    j_min = int(np.argmin(mean))
    h_cvmin = int(hs[j_min])
    threshold = mean[j_min] + se[j_min]
    h_1se = int(hs[np.flatnonzero(mean <= threshold)[0]])
    return h_cvmin, h_1se


def fit_pls(X, y, h):
    return PLSRegression(n_components=int(h), scale=PLS_SCALE).fit(X, y)


# --------------------------------------------------------------------------
# Data (replicates extract_nir_gate_v2.py and run_h1_raman_v2.py)
# --------------------------------------------------------------------------
def load_nir(csv_path):
    from pgsg_1.ingestion import SpectralDataset as DS
    from pgsg_1.ingestion.mango import load_mango_dmc_v3

    import pandas as pd

    full = load_mango_dmc_v3(csv_path)
    mask = full.group_ids == SEASON
    ds4 = DS(X=full.X[mask], y=full.y[mask], wavelengths=full.wavelengths,
             metadata=full.metadata, group_ids=full.group_ids[mask])
    # fruit id = run of consecutive rows with identical metadata and DM
    meta = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp", "DM"]
    s4 = pd.read_csv(csv_path, usecols=meta)
    s4 = s4[s4["Season"] == SEASON].reset_index(drop=True)
    key = s4.astype(str).agg("|".join, axis=1).to_numpy()
    fruit = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1
    if len(s4) != ds4.n_samples or not np.allclose(s4["DM"].to_numpy(), np.asarray(ds4.y).ravel()):
        raise SystemExit("NIR: CSV rows and pgsg_1 loader are not aligned; cannot build fruit groups.")
    return ds4, fruit


def _split_idx(n, groups, split_seed, grouped):
    if grouped:
        return group_split(groups, TEST_FRACTION, split_seed)
    return train_test_split(np.arange(n), test_size=TEST_FRACTION, random_state=split_seed)


def split_nir(ds4, groups, split_seed, grouped):
    """Split, then fit preprocessing on training only (as in the paper)."""
    from pgsg_1.ingestion import SpectralDataset as DS
    from pgsg_1.preprocessing.preprocessor import Preprocessor

    tr, te = _split_idx(ds4.n_samples, groups, split_seed, grouped)
    raw_tr = DS(X=ds4.X[tr], y=ds4.y[tr], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))
    raw_te = DS(X=ds4.X[te], y=ds4.y[te], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))
    pre = Preprocessor(drop_zero_bands=True, apply_snv=True, normalize_target=False)
    X_tr, y_tr = pre.fit_transform(raw_tr)
    X_te, y_te = pre.transform(raw_te)
    wl = pre.params.kept_wavelengths
    return (DS(X=X_tr, y=y_tr, wavelengths=wl, metadata=dict(ds4.metadata)),
            DS(X=X_te, y=y_te, wavelengths=wl, metadata=dict(ds4.metadata)),
            np.asarray(groups)[tr])


def load_raman():
    from pgsg2.ingestion.raman_bioprocess import RamanBioprocessSubstratesLoader
    from pgsg2.preprocessing.raman import RamanPreprocessor
    from pgsg2.models.adapter import to_pgsg1_dataset

    import raman_data

    ds = RamanBioprocessSubstratesLoader().load()
    ds = RamanPreprocessor().fit_transform(ds)  # per-spectrum operations only
    ds1 = to_pgsg1_dataset(ds, target_unit="unknown")
    # mixture id = identical full target vector (same composition)
    raw = raman_data.datasets.load_dataset("bioprocess_substrates")
    T = np.asarray(raw.targets, dtype=float)
    j = [i for i, n in enumerate(raw.target_names) if "glucose" in n.lower()][0]
    valid = ~np.isnan(T[:, j])
    _, mix = np.unique(np.round(np.nan_to_num(T[valid], nan=-1.0), 6), axis=0, return_inverse=True)
    if len(mix) != ds1.n_samples or not np.allclose(T[valid, j], np.asarray(ds1.y).ravel()):
        raise SystemExit("Raman: raw targets and loader are not aligned; cannot build mixture groups.")
    return ds1, mix.ravel()


def split_raman(ds1, groups, split_seed, grouped):
    from pgsg_1.ingestion import SpectralDataset as DS

    tr, te = _split_idx(ds1.n_samples, groups, split_seed, grouped)
    mk = lambda i: DS(X=ds1.X[i], y=ds1.y[i], wavelengths=ds1.wavelengths, metadata=dict(ds1.metadata))
    return mk(tr), mk(te), np.asarray(groups)[tr]


# --------------------------------------------------------------------------
# One repetition
# --------------------------------------------------------------------------
def run_rep(branch, train, test, prior_fn, split_seed, model_seed, args, PGSGv2Model, PLSModel, log,
            train_groups=None):
    rows = []
    Xtr, ytr = np.asarray(train.X), np.asarray(train.y).ravel()
    Xte, yte = np.asarray(test.X), np.asarray(test.y).ravel()
    base = dict(branch=branch, split_seed=split_seed, model_seed=model_seed,
                grouped=train_groups is not None,
                n_train=len(ytr), n_test=len(yte), p=Xtr.shape[1])

    pls10 = PLSModel(n_components=10).fit(train)
    rows.append({**base, "model": "PLS-10", "H": 10,
                 "r2_test": r2_score(yte, np.asarray(pls10.predict(test)).ravel())})

    hs, mean, se = pls_cv_curve(Xtr, ytr, args.h_max, N_FOLDS, split_seed, groups=train_groups)
    h_min, h_1se = select_h(hs, mean, se)
    for name, h in (("PLS-cvmin", h_min), ("PLS-1se", h_1se)):
        m = fit_pls(Xtr, ytr, h)
        rows.append({**base, "model": name, "H": h,
                     "r2_test": r2_score(yte, m.predict(Xte).ravel())})

    prior = prior_fn(train.wavelengths)
    t0 = time.time()
    pg = PGSGv2Model(hidden=args.hidden, max_epochs=args.max_epochs,
                     patience=args.patience, seed=model_seed).fit(train, prior=prior)
    r2_pg = r2_score(yte, np.asarray(pg.predict(test)).ravel())
    rows.append({**base, "model": "PGSGv2", "H": np.nan, "r2_test": r2_pg,
                 "best_epoch": pg.train_history["best_epoch"],
                 "rho_gate_prior": float(np.corrcoef(pg.gates, prior)[0, 1])})
    log(f"  {branch} split={split_seed}: PLS-10={rows[0]['r2_test']:.4f} "
        f"cvmin(H={h_min})={rows[1]['r2_test']:.4f} 1se(H={h_1se})={rows[2]['r2_test']:.4f} "
        f"PGSGv2={r2_pg:.4f} [{time.time()-t0:.0f}s]")
    at_boundary = h_min == int(hs[-1])
    if at_boundary:
        log(f"  WARNING {branch} split={split_seed}: CV minimum at H_max={int(hs[-1])} "
            "-- grid truncated, rerun with larger --h-max")
    sel = dict(branch=branch, split_seed=split_seed, H_cvmin=h_min, H_1se=h_1se,
               H_max=int(hs[-1]), cvmin_at_boundary=bool(at_boundary),
               cv_mse_min=float(mean.min()))
    return rows, sel


def consistency_check(train, test, PLSModel):
    Xtr, ytr = np.asarray(train.X), np.asarray(train.y).ravel()
    Xte, yte = np.asarray(test.X), np.asarray(test.y).ravel()
    r_ref = r2_score(yte, np.asarray(PLSModel(n_components=10).fit(train).predict(test)).ravel())
    r_skl = r2_score(yte, fit_pls(Xtr, ytr, 10).predict(Xte).ravel())
    return r_ref, r_skl


# --------------------------------------------------------------------------
def summarize(runs, n_reps):
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.DataFrame(runs)
    out = {}
    for branch, d in df.groupby("branch"):
        piv = d.pivot_table(index="split_seed", columns="model", values="r2_test")
        res = {"n_reps": int(len(piv)),
               "r2_mean": piv.mean().round(4).to_dict(),
               "r2_sd": piv.std(ddof=1).round(4).to_dict()}
        for ref in ("PLS-10", "PLS-cvmin", "PLS-1se"):
            delta = (piv["PGSGv2"] - piv[ref]).to_numpy()
            p = float(wilcoxon(delta).pvalue) if len(delta) >= 5 and np.any(delta != 0) else float("nan")
            res[f"delta_vs_{ref}"] = {"mean": round(float(delta.mean()), 4),
                                      "sd": round(float(delta.std(ddof=1)), 4) if len(delta) > 1 else None,
                                      "n_positive": int((delta > 0).sum()),
                                      "wilcoxon_p": p}
        d1 = res["delta_vs_PLS-1se"]
        res["H1_decision_vs_PLS-1se"] = ("supported" if d1["mean"] > 0 and d1["wilcoxon_p"] < 0.05
                                         else "not supported")
        out[branch] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pgsg1-root", required=True)
    ap.add_argument("--csv-path", required=True, help="MangoDMC_NIR_Data_v3.csv")
    ap.add_argument("--pgsg2-root", default=str(Path(__file__).resolve().parents[1]),
                    help="pgsg2-nir-raman-gating repo root (default: this script's repo)")
    ap.add_argument("--out-dir", default="results_r1")
    ap.add_argument("--branches", default="nir,raman")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--h-max", type=int, default=H_MAX)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-consistency-check", action="store_true")
    ap.add_argument("--ungrouped", action="store_true",
                    help="random (not sample-grouped) splits for all repetitions")
    args = ap.parse_args()

    n_reps = N_REPS
    if args.quick:
        n_reps, args.max_epochs, args.patience, args.h_max = 2, 30, 10, 15

    # pgsg_v2.py lives at the pgsg_1 root; the pgsg_1 package lives in <root>/src.
    # pgsg2 is taken from --pgsg2-root/src (default: this script's repo).
    root1 = Path(args.pgsg1_root).resolve()
    root2 = Path(args.pgsg2_root).resolve()
    for p in (root1 / "src", root1, root2 / "src"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from pgsg_v2 import PGSGv2Model, make_literature_prior
    from pgsg_1.models.pls import PLSModel
    from pgsg2.priors.raman import RamanGlucosePrior

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = lambda m: print(f"[{time.time()-t_start:8.1f}s] {m}", flush=True)

    branches = {}
    if "nir" in args.branches:
        ds4, g_nir = load_nir(args.csv_path)
        log(f"NIR: {ds4.n_samples} spectra, {len(np.unique(g_nir))} fruits")
        branches["NIR"] = (lambda s, grp: split_nir(ds4, g_nir, s, grp), make_literature_prior)
    if "raman" in args.branches:
        dsr, g_ram = load_raman()
        log(f"Raman: {dsr.n_samples} spectra, {len(np.unique(g_ram))} mixtures")
        branches["Raman"] = (lambda s, grp: split_raman(dsr, g_ram, s, grp),
                             lambda wl: RamanGlucosePrior().compute(wl))

    runs, sels, checks = [], [], {}
    for branch, (splitter, prior_fn) in branches.items():
        log(f"=== {branch} ===")
        tr, te, _ = splitter(42, False)  # reference: original random split
        r_ref, r_skl = consistency_check(tr, te, PLSModel)
        checks[branch] = {"PLSModel_10": r_ref, "sklearn_PLS_10": r_skl}
        log(f"  consistency: PLSModel(10)={r_ref:.6f}  sklearn PLS(10)={r_skl:.6f}")
        if abs(r_ref - r_skl) > 1e-6 and not args.skip_consistency_check:
            raise SystemExit(f"{branch}: sklearn PLS does not match PLSModel -- align "
                             "PLSModel's centering/scaling before interpreting cvmin/1se.")

        rows, sel = run_rep(branch, tr, te, prior_fn, 42, 42, args, PGSGv2Model, PLSModel, log)
        for r in rows: r["repetition"] = "paper"
        runs += rows; sels.append(sel)

        grouped = not args.ungrouped
        for rep in range(n_reps):
            tr, te, g_tr = splitter(rep, grouped)
            rows, sel = run_rep(branch, tr, te, prior_fn, rep, rep, args, PGSGv2Model, PLSModel, log,
                                train_groups=g_tr if grouped else None)
            for r in rows: r["repetition"] = rep
            runs += rows; sels.append(sel)

    import pandas as pd
    pd.DataFrame(runs).to_csv(out / "r1_h1_runs.csv", index=False)
    pd.DataFrame(sels).to_csv(out / "r1_h1_selected_H.csv", index=False)
    summary = {"consistency_check": checks,
               "paper_repetition": pd.DataFrame(runs).query("repetition == 'paper'")
                                     [["branch", "model", "H", "r2_test"]].to_dict("records"),
               "repeated_holdout": summarize([r for r in runs if r["repetition"] != "paper"], n_reps),
               "config": vars(args) | {"n_reps": n_reps, "n_folds": N_FOLDS,
                                       "grouped_repetitions": not args.ungrouped}}
    (out / "r1_h1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"Saved to {out}/")
    print(json.dumps(summary["repeated_holdout"], indent=2, default=float))


if __name__ == "__main__":
    main()
