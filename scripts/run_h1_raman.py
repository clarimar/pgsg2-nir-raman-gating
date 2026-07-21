"""
Script de execução do protocolo experimental H1 (pgsg_2, ramo Raman).

Pipeline: G (ingestão) -> T (pré-processamento) -> P (prior) ->
split treino/teste (intra-domínio, fixo, disjunto) -> adaptador de
contrato -> M (PGSGModel vs PLSModel, ambos de pgsg_1, sem modificação)
-> Delta R^2_PLS.

Uso:
    python run_h1_raman.py --quick     # poucas amostras/bandas/épocas,
                                        # para calibrar tempo antes da
                                        # rodada completa
    python run_h1_raman.py             # configuração completa

O modo --quick subamostra amostras e bandas apenas para estimar o tempo
de execução; os números de R^2 dele NÃO são resultado científico --
servem só para calibração antes da rodada real.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pgsg2.ingestion.raman_bioprocess import RamanBioprocessSubstratesLoader
from pgsg2.preprocessing.raman import RamanPreprocessor
from pgsg2.priors.raman import RamanGlucosePrior
from pgsg2.models.adapter import to_pgsg1_dataset

from pgsg_1.ingestion import SpectralDataset as PGSG1SpectralDataset
from pgsg_1.models.pgsg import PGSGModel
from pgsg_1.models.pls import PLSModel


def _subset_dataset(ds1: PGSG1SpectralDataset, n_samples: int, n_bands: int, seed: int) -> PGSG1SpectralDataset:
    """Subamostra amostras e bandas de um SpectralDataset (pgsg_1),
    preservando os invariantes I1-I4 (wavelengths continua crescente,
    pois é um subconjunto ordenado dos índices originais)."""
    rng = np.random.default_rng(seed)

    n_samples = min(n_samples, ds1.n_samples)
    sample_idx = rng.choice(ds1.n_samples, size=n_samples, replace=False)
    sample_idx.sort()

    n_bands = min(n_bands, ds1.n_bands)
    band_idx = np.linspace(0, ds1.n_bands - 1, n_bands).round().astype(int)
    band_idx = np.unique(band_idx)

    return PGSG1SpectralDataset(
        X=ds1.X[np.ix_(sample_idx, band_idx)],
        y=ds1.y[sample_idx],
        wavelengths=ds1.wavelengths[band_idx],
        metadata=dict(ds1.metadata),
    )


def _split(ds1: PGSG1SpectralDataset, test_frac: float, seed: int):
    idx = np.arange(ds1.n_samples)
    train_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed)
    train = PGSG1SpectralDataset(
        X=ds1.X[train_idx], y=ds1.y[train_idx], wavelengths=ds1.wavelengths,
        metadata=dict(ds1.metadata),
    )
    test = PGSG1SpectralDataset(
        X=ds1.X[test_idx], y=ds1.y[test_idx], wavelengths=ds1.wavelengths,
        metadata=dict(ds1.metadata),
    )
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Protocolo H1 (Raman) -- PGSGv2 vs PLS")
    parser.add_argument("--quick", action="store_true", help="Config rápida para calibrar tempo (NÃO é resultado científico)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-components", type=int, default=10)
    args = parser.parse_args()

    t_start = time.time()

    def log(msg: str):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    # ---------------- G: ingestão ----------------
    log("Carregando bioprocess_substrates via raman-data ...")
    ds2 = RamanBioprocessSubstratesLoader().load()
    log(f"Carregado: n={ds2.n_samples}, p={ds2.n_bands}, dropped={ds2.metadata['n_samples_dropped_missing_target']}")

    # ---------------- T: pré-processamento ----------------
    log("Pré-processando (ALS + cosmic rays + SNV) ...")
    ds2 = RamanPreprocessor().fit_transform(ds2)
    log("Pré-processamento concluído")

    # ---------------- adaptador ----------------
    ds1 = to_pgsg1_dataset(ds2, target_unit="unknown")

    if args.quick:
        log("Modo --quick: subamostrando para calibração de tempo")
        ds1 = _subset_dataset(ds1, n_samples=300, n_bands=200, seed=args.seed)
        max_epochs, patience = 15, 5
    else:
        max_epochs, patience = 200, 20

    log(f"Dataset final: n={ds1.n_samples}, p={ds1.n_bands}")

    # ---------------- split treino/teste ----------------
    train, test = _split(ds1, test_frac=args.test_frac, seed=args.seed)
    log(f"Split: treino n={train.n_samples}, teste n={test.n_samples}")

    # ---------------- P: prior ----------------
    prior = RamanGlucosePrior().compute(train.wavelengths)
    log("Prior de glicose calculado (bandas vibracionais literárias)")

    # ---------------- M: PLS (baseline) ----------------
    log("Treinando PLSModel (baseline) ...")
    t0 = time.time()
    pls = PLSModel(n_components=args.n_components).fit(train)
    log(f"PLSModel treinado em {time.time() - t0:.1f}s")

    # ---------------- M: PGSGModel ----------------
    log(f"Treinando PGSGModel (max_epochs={max_epochs}, patience={patience}) ...")
    t0 = time.time()
    pgsg = PGSGModel(
        n_components=args.n_components, max_epochs=max_epochs, patience=patience, seed=args.seed
    ).fit(train, prior=prior)
    log(f"PGSGModel treinado em {time.time() - t0:.1f}s "
        f"(best_epoch={pgsg.train_history['best_epoch']})")

    # ---------------- avaliação (I_pred) ----------------
    r2_pls_test = r2_score(test.y, pls.predict(test))
    r2_pgsg_test = r2_score(test.y, pgsg.predict(test))
    delta_r2_pls = r2_pgsg_test - r2_pls_test

    print()
    print("=" * 60)
    print(f"R2 PLS  (teste): {r2_pls_test:.4f}")
    print(f"R2 PGSG (teste): {r2_pgsg_test:.4f}")
    print(f"Delta_R2_PLS = R2(PGSGv2) - R2(PLS) = {delta_r2_pls:+.4f}")
    print(f"H1 {'SUPORTADA' if delta_r2_pls > 0 else 'NAO suportada'} nesta execução")
    print("=" * 60)

    # ---------------- preview de topologia do gate (I_interp, informal) ----------------
    g = pgsg.gates
    entropy = -np.sum(g * np.log(g + 1e-300))
    tv = np.mean(np.abs(np.diff(g)))
    hoyer = (np.sqrt(len(g)) - np.linalg.norm(g, 1) / np.linalg.norm(g, 2)) / (np.sqrt(len(g)) - 1)
    print(f"[preview, nao-oficial] Entropia={entropy:.4f}  TV={tv:.6f}  Hoyer={hoyer:.4f}")
    print("(cálculo formal via operador I_interp fica para a etapa 7-8 do cronograma)")

    log("Concluído.")


if __name__ == "__main__":
    main()
