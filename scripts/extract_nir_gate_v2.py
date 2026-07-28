"""
Extração do gate NIR (Mango DMC v3) para comparação de topologia com o
ramo Raman de pgsg_2 (H2/H3/H5) -- VERSÃO CORRIGIDA.

Usa PGSGv2Model (sigmoid + MLP + Adam) e make_literature_prior, ambos
de pgsg_v2.py -- o modelo e o prior que de fato geraram os resultados
publicados de pgsg_1 (results_v2/). A versão anterior deste script
(extract_nir_gate.py) usava PGSGModel + VIPPrior, uma combinação nunca
utilizada para os resultados finais do artigo -- ver correção
registrada em memória/ADR.

Replica o protocolo real de run_experiment_v2.py: Mango DMC v3, Safra 4
apenas (n=1.448), teste = 20% fixo estratificado por y (seed=42), treino
= resto (n≈1.159). Preprocessor com drop_zero_bands=True, apply_snv=True,
normalize_target=False (y NÃO normalizado, ao contrário do que eu havia
assumido antes -- ver run_experiment_v2.py, linha do Preprocessor).

pgsg_v2.py NÃO faz parte do pacote instalável pgsg_1 (é um arquivo solto
na raiz do repositório, usado via sys.path por run_experiment_v2.py).
Este script replica exatamente essa forma de importação, sem modificar
a estrutura de pgsg_1.

Uso:
    python scripts/extract_nir_gate_v2.py \\
        --pgsg1-root /home/clarimar/Dropbox/pgsg/pgsg_1 \\
        --csv-path /home/clarimar/Dropbox/pgsg/pgsg_1/data/mango_dmc_v3/MangoDMC_NIR_Data_v3.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pgsg_1.ingestion import SpectralDataset as PGSG1SpectralDataset
from pgsg_1.ingestion.mango import load_mango_dmc_v3
from pgsg_1.preprocessing.preprocessor import Preprocessor
from pgsg_1.models.pls import PLSModel

from pgsg2.interpretability.topology import gate_topology

SEASON = 4  # ver run_experiment_v2.py: DATA_PATH, RESULTS_DIR, SEASON = 4
TEST_FRACTION = 0.2
SPLIT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Extrai o gate NIR (Mango DMC v3, Safra 4) via PGSGv2Model, sem modificação")
    parser.add_argument("--pgsg1-root", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n-components-pls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="nir_gate_v2_extraction.npz")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.pgsg1_root).resolve()))
    from pgsg_v2 import PGSGv2Model, make_literature_prior  # import tardio

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    log(f"Carregando Mango DMC v3, Safra {SEASON} (protocolo real de run_experiment_v2.py) ...")
    ds_full = load_mango_dmc_v3(args.csv_path)
    mask = ds_full.group_ids == SEASON
    ds4 = PGSG1SpectralDataset(
        X=ds_full.X[mask], y=ds_full.y[mask],
        wavelengths=ds_full.wavelengths, metadata=ds_full.metadata,
        group_ids=ds_full.group_ids[mask],
    )
    log(f"Safra {SEASON}: n={ds4.n_samples}, p_raw={ds4.n_bands}")

    # split 80/20 estratificado por y, seed=42 (replicando FixedFractionTest)
    idx = np.arange(ds4.n_samples)
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRACTION, random_state=SPLIT_SEED,
    )
    ds_train_raw = PGSG1SpectralDataset(
        X=ds4.X[train_idx], y=ds4.y[train_idx], wavelengths=ds4.wavelengths,
        metadata=dict(ds4.metadata),
    )
    ds_test_raw = PGSG1SpectralDataset(
        X=ds4.X[test_idx], y=ds4.y[test_idx], wavelengths=ds4.wavelengths,
        metadata=dict(ds4.metadata),
    )
    log(f"Split: treino n={ds_train_raw.n_samples}, teste n={ds_test_raw.n_samples}")

    log("Pré-processando (drop_zero_bands + SNV; y NÃO normalizado, replicando run_experiment_v2.py) ...")
    pre = Preprocessor(drop_zero_bands=True, apply_snv=True, normalize_target=False)
    X_tr, y_tr = pre.fit_transform(ds_train_raw)
    X_te, y_te = pre.transform(ds_test_raw)
    kept_wl = pre.params.kept_wavelengths
    log(f"Pré-processado: p_mantido={X_tr.shape[1]}")

    train_ds = PGSG1SpectralDataset(X=X_tr, y=y_tr, wavelengths=kept_wl, metadata=dict(ds4.metadata))
    test_ds = PGSG1SpectralDataset(X=X_te, y=y_te, wavelengths=kept_wl, metadata=dict(ds4.metadata))

    prior_lit = make_literature_prior(kept_wl)
    log(f"Prior de literatura construído (bandas altas: {(prior_lit > 0.8).sum()})")

    log("Treinando PLSModel (baseline) ...")
    t_pls = time.time()
    pls = PLSModel(n_components=args.n_components_pls).fit(train_ds)
    log(f"PLSModel treinado em {time.time()-t_pls:.1f}s")

    log(f"Treinando PGSGv2Model (hidden={args.hidden}, max_epochs={args.max_epochs}, patience={args.patience}) ...")
    t_pgsg = time.time()
    model = PGSGv2Model(
        hidden=args.hidden, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed,
    ).fit(train_ds, prior=prior_lit)
    log(f"PGSGv2Model treinado em {time.time()-t_pgsg:.1f}s (best_epoch={model.train_history['best_epoch']})")

    r2_pls = r2_score(test_ds.y, pls.predict(test_ds))
    r2_pgsg = r2_score(test_ds.y, model.predict(test_ds))

    gate = model.gates
    prior_used = model.prior_used
    rho = float(np.corrcoef(gate, prior_used)[0, 1])
    topo = gate_topology(gate)

    print()
    print("=" * 60)
    print(f"R2 PLS    (Safra 4, teste 20%): {r2_pls:.4f}")
    print(f"R2 PGSGv2 (Safra 4, teste 20%): {r2_pgsg:.4f}")
    print(f"Delta_R2_PLS = {r2_pgsg - r2_pls:+.4f}")
    print(f"best_epoch = {model.train_history['best_epoch']}")
    print(f"rho(gate, prior) = {rho:.6f}")
    print(f"Topologia do gate NIR: entropy={topo['entropy']:.4f}  "
          f"smoothness={topo['smoothness']:.6f}  sparsity_hoyer={topo['sparsity_hoyer']:.4f}")
    print("=" * 60)

    np.savez(
        args.out,
        gate=gate, prior_used=prior_used, wavelengths=kept_wl,
        r2_pls=r2_pls, r2_pgsg=r2_pgsg,
        best_epoch=model.train_history["best_epoch"],
        train_losses=model.train_history["train_losses"],
        val_losses=model.train_history["val_losses"],
        rho_gate_prior=rho, **topo,
        n_train=train_ds.n_samples, n_test=test_ds.n_samples, n_bands=train_ds.n_bands,
    )
    log(f"Gate, prior e topologia salvos em {args.out}")


if __name__ == "__main__":
    main()
