"""
Protocolo experimental H4 (pgsg_2): vantagem do prior fisicamente
motivado.

Hipótese: a inicialização do gate por atribuições vibracionais da
literatura produz desempenho preditivo igual ou superior, e maior
estabilidade do gate entre sementes, em comparação com inicialização
não informada (prior=None -- theta inicia em zeros, gate uniforme
0.5), em ambas as modalidades (NIR e Raman).

Replica, em espírito, a ablação "PGSGv2-random" de run_experiment_v2.py
(prior=None), mas com múltiplas sementes em ambas as condições para
medir estabilidade (não só desempenho pontual).

Uso:
    python scripts/run_h4_prior_ablation.py \\
        --pgsg1-root /home/clarimar/Dropbox/pgsg/pgsg_1 \\
        --csv-path /home/clarimar/Dropbox/pgsg/pgsg_1/data/mango_dmc_v3/MangoDMC_NIR_Data_v3.csv \\
        --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import itertools
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


def _top_q_indices(g: np.ndarray, q: int) -> set:
    return set(np.argsort(g)[-q:].tolist())


def _jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 1.0


def _pairwise_stats(gates: list[np.ndarray], top_q: int) -> dict:
    """Correlação e Jaccard médios entre todos os pares de sementes."""
    rhos, jaccards = [], []
    for g_a, g_b in itertools.combinations(gates, 2):
        rhos.append(float(np.corrcoef(g_a, g_b)[0, 1]))
        jaccards.append(_jaccard(_top_q_indices(g_a, top_q), _top_q_indices(g_b, top_q)))
    return {
        "rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "rho_std": float(np.std(rhos)) if rhos else float("nan"),
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else float("nan"),
        "jaccard_std": float(np.std(jaccards)) if jaccards else float("nan"),
    }


def _run_condition(model_cls, train, test, prior, seeds, hidden, max_epochs, patience):
    r2s, gates = [], []
    for seed in seeds:
        model = model_cls(hidden=hidden, max_epochs=max_epochs, patience=patience, seed=seed).fit(train, prior=prior)
        r2s.append(r2_score(test.y, model.predict(test)))
        gates.append(model.gates)
    return np.array(r2s), gates


def main():
    parser = argparse.ArgumentParser(description="H4: prior de literatura vs. init não-informada, multi-seed")
    parser.add_argument("--pgsg1-root", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    sys.path.insert(0, str(Path(args.pgsg1_root).resolve()))
    from pgsg_v2 import PGSGv2Model, make_literature_prior

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    results = {}

    # ================= NIR =================
    log("=== NIR (Mango DMC v3, Safra 4) ===")
    ds_full = load_mango_dmc_v3(args.csv_path)
    mask = ds_full.group_ids == SEASON
    ds4 = PGSG1SpectralDataset(
        X=ds_full.X[mask], y=ds_full.y[mask], wavelengths=ds_full.wavelengths,
        metadata=ds_full.metadata, group_ids=ds_full.group_ids[mask],
    )
    idx = np.arange(ds4.n_samples)
    train_idx, test_idx = train_test_split(idx, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    ds_train_raw = PGSG1SpectralDataset(X=ds4.X[train_idx], y=ds4.y[train_idx], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))
    ds_test_raw = PGSG1SpectralDataset(X=ds4.X[test_idx], y=ds4.y[test_idx], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))

    pre_nir = Preprocessor(drop_zero_bands=True, apply_snv=True, normalize_target=False)
    X_tr, y_tr = pre_nir.fit_transform(ds_train_raw)
    X_te, y_te = pre_nir.transform(ds_test_raw)
    kept_wl_nir = pre_nir.params.kept_wavelengths
    train_nir = PGSG1SpectralDataset(X=X_tr, y=y_tr, wavelengths=kept_wl_nir, metadata=dict(ds4.metadata))
    test_nir = PGSG1SpectralDataset(X=X_te, y=y_te, wavelengths=kept_wl_nir, metadata=dict(ds4.metadata))
    prior_nir = make_literature_prior(kept_wl_nir)
    top_q_nir = max(1, len(kept_wl_nir) // 10)

    log(f"Treinando NIR com prior de literatura, seeds={seeds} ...")
    r2_lit_nir, gates_lit_nir = _run_condition(PGSGv2Model, train_nir, test_nir, prior_nir, seeds, args.hidden, args.max_epochs, args.patience)
    log(f"Treinando NIR com init não-informada (prior=None), seeds={seeds} ...")
    r2_rand_nir, gates_rand_nir = _run_condition(PGSGv2Model, train_nir, test_nir, None, seeds, args.hidden, args.max_epochs, args.patience)

    stab_lit_nir = _pairwise_stats(gates_lit_nir, top_q_nir)
    stab_rand_nir = _pairwise_stats(gates_rand_nir, top_q_nir)

    results["nir"] = {
        "r2_lit": r2_lit_nir, "r2_rand": r2_rand_nir,
        "stab_lit": stab_lit_nir, "stab_rand": stab_rand_nir,
    }
    log(f"NIR lit : R2={r2_lit_nir.mean():.4f}+-{r2_lit_nir.std():.4f}  rho_pairwise={stab_lit_nir['rho_mean']:.4f}  jaccard={stab_lit_nir['jaccard_mean']:.4f}")
    log(f"NIR rand: R2={r2_rand_nir.mean():.4f}+-{r2_rand_nir.std():.4f}  rho_pairwise={stab_rand_nir['rho_mean']:.4f}  jaccard={stab_rand_nir['jaccard_mean']:.4f}")

    # ================= Raman =================
    log("=== Raman (bioprocess_substrates) ===")
    ds2 = RamanBioprocessSubstratesLoader().load()
    ds2 = RamanPreprocessor().fit_transform(ds2)
    ds1_raman = to_pgsg1_dataset(ds2, target_unit="unknown")
    idx_r = np.arange(ds1_raman.n_samples)
    train_idx_r, test_idx_r = train_test_split(idx_r, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    train_raman = PGSG1SpectralDataset(X=ds1_raman.X[train_idx_r], y=ds1_raman.y[train_idx_r], wavelengths=ds1_raman.wavelengths, metadata=dict(ds1_raman.metadata))
    test_raman = PGSG1SpectralDataset(X=ds1_raman.X[test_idx_r], y=ds1_raman.y[test_idx_r], wavelengths=ds1_raman.wavelengths, metadata=dict(ds1_raman.metadata))
    prior_raman = RamanGlucosePrior().compute(train_raman.wavelengths)
    top_q_raman = max(1, train_raman.n_bands // 10)

    log(f"Treinando Raman com prior de literatura, seeds={seeds} ...")
    r2_lit_raman, gates_lit_raman = _run_condition(PGSGv2Model, train_raman, test_raman, prior_raman, seeds, args.hidden, args.max_epochs, args.patience)
    log(f"Treinando Raman com init não-informada (prior=None), seeds={seeds} ...")
    r2_rand_raman, gates_rand_raman = _run_condition(PGSGv2Model, train_raman, test_raman, None, seeds, args.hidden, args.max_epochs, args.patience)

    stab_lit_raman = _pairwise_stats(gates_lit_raman, top_q_raman)
    stab_rand_raman = _pairwise_stats(gates_rand_raman, top_q_raman)

    results["raman"] = {
        "r2_lit": r2_lit_raman, "r2_rand": r2_rand_raman,
        "stab_lit": stab_lit_raman, "stab_rand": stab_rand_raman,
    }
    log(f"Raman lit : R2={r2_lit_raman.mean():.4f}+-{r2_lit_raman.std():.4f}  rho_pairwise={stab_lit_raman['rho_mean']:.4f}  jaccard={stab_lit_raman['jaccard_mean']:.4f}")
    log(f"Raman rand: R2={r2_rand_raman.mean():.4f}+-{r2_rand_raman.std():.4f}  rho_pairwise={stab_rand_raman['rho_mean']:.4f}  jaccard={stab_rand_raman['jaccard_mean']:.4f}")

    # ================= resumo =================
    print()
    print("=" * 78)
    print("RESUMO H4: prior de literatura vs. inicialização não-informada")
    print("=" * 78)
    for modality, r in results.items():
        print(f"\n[{modality.upper()}]")
        print(f"  R2       lit={r['r2_lit'].mean():.4f}+-{r['r2_lit'].std():.4f}   "
              f"rand={r['r2_rand'].mean():.4f}+-{r['r2_rand'].std():.4f}   "
              f"{'lit >= rand' if r['r2_lit'].mean() >= r['r2_rand'].mean() else 'rand > lit'}")
        print(f"  rho(pares)   lit={r['stab_lit']['rho_mean']:.4f}   rand={r['stab_rand']['rho_mean']:.4f}   "
              f"{'lit mais estavel' if r['stab_lit']['rho_mean'] >= r['stab_rand']['rho_mean'] else 'rand mais estavel'}")
        print(f"  jaccard(pares) lit={r['stab_lit']['jaccard_mean']:.4f}   rand={r['stab_rand']['jaccard_mean']:.4f}   "
              f"{'lit mais estavel' if r['stab_lit']['jaccard_mean'] >= r['stab_rand']['jaccard_mean'] else 'rand mais estavel'}")
    print("=" * 78)

    np.savez(
        "h4_prior_ablation_result.npz",
        r2_lit_nir=r2_lit_nir, r2_rand_nir=r2_rand_nir,
        r2_lit_raman=r2_lit_raman, r2_rand_raman=r2_rand_raman,
        rho_lit_nir=stab_lit_nir["rho_mean"], rho_rand_nir=stab_rand_nir["rho_mean"],
        rho_lit_raman=stab_lit_raman["rho_mean"], rho_rand_raman=stab_rand_raman["rho_mean"],
        jaccard_lit_nir=stab_lit_nir["jaccard_mean"], jaccard_rand_nir=stab_rand_nir["jaccard_mean"],
        jaccard_lit_raman=stab_lit_raman["jaccard_mean"], jaccard_rand_raman=stab_rand_raman["jaccard_mean"],
        seeds=np.array(seeds),
    )
    log("Resultado salvo em h4_prior_ablation_result.npz")


if __name__ == "__main__":
    main()
