"""
R1 diagnostic: are there replicate spectra of the same physical sample?

If a sample (mango fruit, bioprocess mixture) contributes more than one
spectrum, a random 80/20 split places replicates in both partitions, and
random K-fold CV does the same inside the training set. Test R2 is then
optimistic, and CV rewards models that memorize samples (e.g. very large
PLS H). Splits must then be grouped by sample.

NIR  : Mango DMC v3 has no fruit-ID column. Candidate groups are runs of
       consecutive rows with identical metadata and DM. The script checks
       (i) run-length distribution in Season 4, (ii) alignment with the
       pgsg_1 loader (same n, same y order), and (iii) whether spectra within
       a run are more similar than spectra of different fruits.
Raman: candidate groups are rows with identical full target vectors
       (all substrate concentrations), i.e. the same mixture.

Usage:
    python scripts/r1_check_replicates.py \
        --pgsg1-root /home/clarimar/Dropbox/pgsg/pgsg_1 \
        --csv-path /home/clarimar/Dropbox/pgsg/pgsg_1/data/mango_dmc_v3/MangoDMC_NIR_Data_v3.csv \
        --out /home/clarimar/pgsg_results/pgsg_2/r1/replicates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = 4
META = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp", "DM"]


def run_ids(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Consecutive rows with identical values in `cols` share an id."""
    key = df[cols].astype(str).agg("|".join, axis=1).to_numpy()
    new = np.r_[True, key[1:] != key[:-1]]
    return np.cumsum(new) - 1


def spectral_similarity(X: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """Median correlation within groups of size 2 vs. random pairs from different groups."""
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    within = []
    for g, c in Counter(groups).items():
        if c == 2:
            i, j = np.flatnonzero(groups == g)
            within.append(float(Xc[i] @ Xc[j]))
    n = len(groups)
    a, b = rng.integers(0, n, 5000), rng.integers(0, n, 5000)
    keep = groups[a] != groups[b]
    between = (Xc[a[keep]] * Xc[b[keep]]).sum(axis=1)
    return {"within_pairs_n": len(within),
            "within_median_r": float(np.median(within)) if within else None,
            "between_median_r": float(np.median(between)),
            "between_p99_r": float(np.quantile(between, 0.99))}


def check_nir(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    s4 = df[df["Season"] == SEASON].reset_index(drop=True)
    wl_cols = [c for c in df.columns if c not in META]
    gid = run_ids(s4, META)
    sizes = Counter(Counter(gid).values())
    out = {"n_rows_season4": int(len(s4)), "n_groups": int(gid.max() + 1),
           "group_size_counts": {int(k): int(v) for k, v in sorted(sizes.items())},
           "similarity": spectral_similarity(s4[wl_cols].to_numpy(float), gid)}

    from pgsg_1.ingestion.mango import load_mango_dmc_v3
    ds = load_mango_dmc_v3(csv_path)
    y4 = np.asarray(ds.y)[np.asarray(ds.group_ids) == SEASON]
    out["loader_n_season4"] = int(len(y4))
    out["loader_order_matches_csv"] = bool(len(y4) == len(s4) and np.allclose(y4, s4["DM"].to_numpy()))
    return out


def check_raman() -> dict:
    import raman_data

    raw = raman_data.datasets.load_dataset("bioprocess_substrates")
    T = np.asarray(raw.targets, dtype=float)
    names = list(raw.target_names)
    j = [i for i, n in enumerate(names) if "glucose" in n.lower()][0]
    valid = ~np.isnan(T[:, j])
    Tv = np.round(np.nan_to_num(T[valid], nan=-1.0), 6)
    _, gid = np.unique(Tv, axis=0, return_inverse=True)
    gid = gid.ravel()
    sizes = Counter(Counter(gid).values())
    extra = {a: str(type(getattr(raw, a)).__name__) for a in dir(raw)
             if not a.startswith("_") and a not in ("spectra", "targets", "raman_shifts", "target_names")}
    return {"target_names": names, "n_valid_glucose": int(valid.sum()),
            "n_unique_target_vectors": int(gid.max() + 1),
            "n_unique_glucose_values": int(len(np.unique(np.round(T[valid, j], 6)))),
            "group_size_counts": {int(k): int(v) for k, v in sorted(sizes.items())},
            "similarity": spectral_similarity(np.asarray(raw.spectra, float)[valid], gid),
            "other_attributes": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgsg1-root", required=True)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--pgsg2-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="replicates.json")
    args = ap.parse_args()
    for p in (Path(args.pgsg1_root) / "src", Path(args.pgsg1_root), Path(args.pgsg2_root) / "src"):
        if p.is_dir():
            sys.path.insert(0, str(p.resolve()))

    res = {"NIR": check_nir(args.csv_path), "Raman": check_raman()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
