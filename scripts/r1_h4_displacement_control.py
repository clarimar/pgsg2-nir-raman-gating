"""
H4 control (pgsg_2 revision): does the literature prior stabilise what the
gate *learns*, or does it merely stop the gate from moving?

Both initialisations in run_h4_prior_ablation.py are deterministic and
identical across seeds: the literature prior is a function of the spectral
axis alone, and the uninformed condition sets theta = 0, so g = 0.5 for every
variable. The pairwise gate correlation reported for H4 therefore does not
compare like with like. Under the prior, the correlation is dominated by a
structure that is the same in every run by construction; under the uninformed
initialisation the starting vector is constant, so its correlation is computed
entirely on what training produced.

This script separates the two by measuring the displacement

    d = g_final - g_init,       g_init = prior,  or 0.5 * ones

and asking whether d, rather than g, agrees across seeds. It also reports how
far the gate travels at all, and how much of it is pinned by saturation of the
logit transform: a prior value of exactly 1.0 (both priors are max-normalised)
is clipped to 1 - 1e-6 and maps to theta = 13.8155, where dg/dtheta = 1e-6,
against 0.25 at g = 0.5.

Interpretation:
  - If rho(d) under the prior is high, the prior stabilises a genuine learned
    pattern and the H4 claim stands as written.
  - If rho(d) is low while rho(g) is high, the reported stability is the
    stability of the initialisation, not of anything learned.

Usage mirrors run_h4_prior_ablation.py:

    python scripts/r1_h4_displacement_control.py \
        --pgsg1-root /path/to/pgsg_1 \
        --csv-path   /path/to/MangoDMC_NIR_Data_v3.csv \
        --seeds 0,1,2,3,4,5,6,7,8,9
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pgsg_1.ingestion import SpectralDataset as PGSG1SpectralDataset
from pgsg_1.ingestion.mango import load_mango_dmc_v3
from pgsg_1.preprocessing.preprocessor import Preprocessor

from pgsg2.ingestion.raman_bioprocess import RamanBioprocessSubstratesLoader
from pgsg2.preprocessing.raman import RamanPreprocessor
from pgsg2.priors.raman import RamanGlucosePrior
from pgsg2.models.adapter import to_pgsg1_dataset

SEASON = 4
TEST_FRACTION = 0.2
SPLIT_SEED = 42
EPS = 1e-6
SATURATION_LOGIT = 10.0  # |theta| above which dg/dtheta < 5e-5


def _top_q_indices(v: np.ndarray, q: int) -> set:
    return set(np.argsort(v)[-q:].tolist())


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def _pairwise(vectors: list[np.ndarray], q: int) -> dict:
    """Pairwise correlation and top-decile Jaccard over a list of vectors."""
    rhos, jacs = [], []
    for a, b in itertools.combinations(vectors, 2):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            rhos.append(float("nan"))
        else:
            rhos.append(float(np.corrcoef(a, b)[0, 1]))
        jacs.append(_jaccard(_top_q_indices(a, q), _top_q_indices(b, q)))
    return {
        "rho_mean": float(np.nanmean(rhos)),
        "rho_std": float(np.nanstd(rhos)),
        "jaccard_mean": float(np.mean(jacs)),
        "jaccard_std": float(np.std(jacs)),
    }


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), EPS, 1 - EPS)
    return np.log(s / (1 - s))


def _init_gate(prior: np.ndarray | None, p: int) -> np.ndarray:
    """The gate the model starts from, before any training step."""
    if prior is None:
        return np.full(p, 0.5)
    return 1.0 / (1.0 + np.exp(-_logit(prior)))


def _run_condition(model_cls, train, test, prior, seeds, hidden, max_epochs,
                   patience):
    r2s, gates = [], []
    for seed in seeds:
        model = model_cls(hidden=hidden, max_epochs=max_epochs,
                          patience=patience, seed=seed).fit(train, prior=prior)
        r2s.append(float(r2_score(test.y, model.predict(test))))
        gates.append(np.asarray(model.gates, dtype=np.float64))
    return np.array(r2s), gates


def _analyse(name, prior, gates, r2s, q):
    """Compare stability of the final gate against stability of the movement."""
    p = len(gates[0])
    g0 = _init_gate(prior, p)
    disps = [g - g0 for g in gates]

    stab_gate = _pairwise(gates, q)
    stab_disp = _pairwise(disps, q)

    travel = np.array([float(np.abs(d).max()) for d in disps])
    travel_mean = np.array([float(np.abs(d).mean()) for d in disps])
    rho_to_init = np.array([
        float(np.corrcoef(g, g0)[0, 1]) if np.std(g0) > 1e-12 else float("nan")
        for g in gates
    ])

    theta0 = _logit(prior) if prior is not None else np.zeros(p)
    saturated = float(np.mean(np.abs(theta0) > SATURATION_LOGIT))
    slope = 1.0 / (1.0 + np.exp(-theta0))
    slope = slope * (1 - slope)

    print(f"\n  --- {name} ---")
    print(f"  test R2                      {r2s.mean():.4f} +/- {r2s.std():.4f}")
    print(f"  gate: pairwise rho           {stab_gate['rho_mean']:.4f}"
          f" +/- {stab_gate['rho_std']:.4f}")
    print(f"  gate: pairwise Jaccard       {stab_gate['jaccard_mean']:.4f}")
    print(f"  DISPLACEMENT: pairwise rho   {stab_disp['rho_mean']:.4f}"
          f" +/- {stab_disp['rho_std']:.4f}    <== the control")
    print(f"  DISPLACEMENT: pairwise Jacc. {stab_disp['jaccard_mean']:.4f}")
    print(f"  rho(final gate, init)        {rho_to_init.mean():.4f}")
    print(f"  |displacement| max           {travel.mean():.4f}"
          f"  [{travel.min():.4f}; {travel.max():.4f}]")
    print(f"  |displacement| mean          {travel_mean.mean():.6f}")
    print(f"  variables saturated at init  {100*saturated:.1f}%"
          f"   (|theta0| > {SATURATION_LOGIT:g})")
    print(f"  median dg/dtheta at init     {np.median(slope):.3e}"
          f"   (0.25 at g = 0.5)")

    return {
        "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
        "gate": stab_gate, "displacement": stab_disp,
        "rho_to_init": float(rho_to_init.mean()),
        "travel_max": float(travel.mean()),
        "travel_mean": float(travel_mean.mean()),
        "frac_saturated": saturated,
        "median_slope": float(np.median(slope)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H4 control: stability of the learned displacement")
    parser.add_argument("--pgsg1-root", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--out", type=str, default="revision_r1/r1_h4_displacement.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    sys.path.insert(0, str(Path(args.pgsg1_root).resolve()))
    from pgsg_v2 import PGSGv2Model, make_literature_prior

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    out = {"seeds": seeds}

    # ---------------- NIR ----------------
    log("=== NIR (Mango DMC v3, Season 4) ===")
    ds_full = load_mango_dmc_v3(args.csv_path)
    mask = ds_full.group_ids == SEASON
    ds4 = PGSG1SpectralDataset(
        X=ds_full.X[mask], y=ds_full.y[mask], wavelengths=ds_full.wavelengths,
        metadata=ds_full.metadata, group_ids=ds_full.group_ids[mask],
    )
    idx = np.arange(ds4.n_samples)
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRACTION,
                                      random_state=SPLIT_SEED)
    raw_tr = PGSG1SpectralDataset(X=ds4.X[tr_idx], y=ds4.y[tr_idx],
                                  wavelengths=ds4.wavelengths,
                                  metadata=dict(ds4.metadata))
    raw_te = PGSG1SpectralDataset(X=ds4.X[te_idx], y=ds4.y[te_idx],
                                  wavelengths=ds4.wavelengths,
                                  metadata=dict(ds4.metadata))
    pre = Preprocessor(drop_zero_bands=True, apply_snv=True,
                       normalize_target=False)
    X_tr, y_tr = pre.fit_transform(raw_tr)
    X_te, y_te = pre.transform(raw_te)
    wl = pre.params.kept_wavelengths
    train_nir = PGSG1SpectralDataset(X=X_tr, y=y_tr, wavelengths=wl,
                                     metadata=dict(ds4.metadata))
    test_nir = PGSG1SpectralDataset(X=X_te, y=y_te, wavelengths=wl,
                                    metadata=dict(ds4.metadata))
    prior_nir = make_literature_prior(wl)
    q_nir = max(1, len(wl) // 10)

    log(f"NIR, literature prior, seeds={seeds} ...")
    r2_lit, g_lit = _run_condition(PGSGv2Model, train_nir, test_nir, prior_nir,
                                   seeds, args.hidden, args.max_epochs,
                                   args.patience)
    log("NIR, uninformed initialisation ...")
    r2_rnd, g_rnd = _run_condition(PGSGv2Model, train_nir, test_nir, None,
                                   seeds, args.hidden, args.max_epochs,
                                   args.patience)

    print("\n" + "=" * 74)
    print(" NIR (Mango DMC v3, Season 4)")
    print("=" * 74)
    out["nir_literature"] = _analyse("literature prior", prior_nir, g_lit,
                                     r2_lit, q_nir)
    out["nir_uninformed"] = _analyse("uninformed init", None, g_rnd,
                                     r2_rnd, q_nir)

    # ---------------- Raman ----------------
    log("=== Raman (bioprocess_substrates) ===")
    ds2 = RamanPreprocessor().fit_transform(
        RamanBioprocessSubstratesLoader().load())
    ds1 = to_pgsg1_dataset(ds2, target_unit="unknown")
    idx_r = np.arange(ds1.n_samples)
    tr_r, te_r = train_test_split(idx_r, test_size=TEST_FRACTION,
                                  random_state=SPLIT_SEED)
    train_ram = PGSG1SpectralDataset(X=ds1.X[tr_r], y=ds1.y[tr_r],
                                     wavelengths=ds1.wavelengths,
                                     metadata=dict(ds1.metadata))
    test_ram = PGSG1SpectralDataset(X=ds1.X[te_r], y=ds1.y[te_r],
                                    wavelengths=ds1.wavelengths,
                                    metadata=dict(ds1.metadata))
    prior_ram = RamanGlucosePrior().compute(train_ram.wavelengths)
    q_ram = max(1, train_ram.n_bands // 10)

    log(f"Raman, literature prior, seeds={seeds} ...")
    r2_lit_r, g_lit_r = _run_condition(PGSGv2Model, train_ram, test_ram,
                                       prior_ram, seeds, args.hidden,
                                       args.max_epochs, args.patience)
    log("Raman, uninformed initialisation ...")
    r2_rnd_r, g_rnd_r = _run_condition(PGSGv2Model, train_ram, test_ram, None,
                                       seeds, args.hidden, args.max_epochs,
                                       args.patience)

    print("\n" + "=" * 74)
    print(" RAMAN (bioprocess_substrates)")
    print("=" * 74)
    out["raman_literature"] = _analyse("literature prior", prior_ram, g_lit_r,
                                       r2_lit_r, q_ram)
    out["raman_uninformed"] = _analyse("uninformed init", None, g_rnd_r,
                                       r2_rnd_r, q_ram)

    # ---------------- summary ----------------
    print("\n" + "=" * 74)
    print(" SUMMARY: final gate versus learned displacement")
    print("=" * 74)
    print(f"  {'condition':<28}{'rho(gate)':>12}{'rho(displacement)':>20}")
    for key, label in (
        ("nir_literature", "NIR, literature prior"),
        ("nir_uninformed", "NIR, uninformed"),
        ("raman_literature", "Raman, literature prior"),
        ("raman_uninformed", "Raman, uninformed"),
    ):
        r = out[key]
        print(f"  {label:<28}{r['gate']['rho_mean']:>12.4f}"
              f"{r['displacement']['rho_mean']:>20.4f}")
    print()
    print("  If rho(gate) is high under the prior while rho(displacement) is")
    print("  not, the stability reported for H4 is the stability of the")
    print("  initialisation, not of a learned pattern.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
