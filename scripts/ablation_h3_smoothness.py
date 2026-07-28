"""
Ablação para H3 (suavidade do gate): resolve o confundidor de forma
funcional do prior (degrau em NIR vs. gaussiana em Raman).

Duas verificações independentes:
    (a) Re-treina o NIR com um prior GAUSSIANO (mesmos centros/pesos de
        make_literature_prior, forma suave em vez de degrau) e recalcula
        a topologia do gate resultante -- comparável em estilo ao prior
        Raman.
    (b) Calcula a suavidade do ESPECTRO BRUTO pré-processado (não do
        gate/prior) em ambas as modalidades -- evidência independente de
        qualquer escolha de prior.

Uso:
    python scripts/ablation_h3_smoothness.py \\
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

from pgsg2.ingestion.raman_bioprocess import RamanBioprocessSubstratesLoader
from pgsg2.preprocessing.raman import RamanPreprocessor
from pgsg2.priors.raman import RamanGlucosePrior
from pgsg2.priors.nir_gaussian import NIRLiteratureGaussianPrior
from pgsg2.models.adapter import to_pgsg1_dataset
from pgsg2.interpretability.topology import gate_topology, spectrum_smoothness

SEASON = 4
TEST_FRACTION = 0.2
SPLIT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Ablação H3: prior gaussiano NIR + suavidade do espectro bruto")
    parser.add_argument("--pgsg1-root", type=str, required=True)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n-components-pls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.pgsg1_root).resolve()))
    from pgsg_v2 import PGSGv2Model

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    # ================= (a) NIR com prior gaussiano =================
    log("--- Parte (a): NIR com prior gaussiano (ablação de forma funcional) ---")
    ds_full = load_mango_dmc_v3(args.csv_path)
    mask = ds_full.group_ids == SEASON
    ds4 = PGSG1SpectralDataset(
        X=ds_full.X[mask], y=ds_full.y[mask],
        wavelengths=ds_full.wavelengths, metadata=ds_full.metadata,
        group_ids=ds_full.group_ids[mask],
    )
    idx = np.arange(ds4.n_samples)
    train_idx, test_idx = train_test_split(idx, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    ds_train_raw = PGSG1SpectralDataset(X=ds4.X[train_idx], y=ds4.y[train_idx], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))
    ds_test_raw = PGSG1SpectralDataset(X=ds4.X[test_idx], y=ds4.y[test_idx], wavelengths=ds4.wavelengths, metadata=dict(ds4.metadata))

    pre_nir = Preprocessor(drop_zero_bands=True, apply_snv=True, normalize_target=False)
    X_tr_nir, y_tr_nir = pre_nir.fit_transform(ds_train_raw)
    X_te_nir, y_te_nir = pre_nir.transform(ds_test_raw)
    kept_wl_nir = pre_nir.params.kept_wavelengths

    train_ds_nir = PGSG1SpectralDataset(X=X_tr_nir, y=y_tr_nir, wavelengths=kept_wl_nir, metadata=dict(ds4.metadata))
    test_ds_nir = PGSG1SpectralDataset(X=X_te_nir, y=y_te_nir, wavelengths=kept_wl_nir, metadata=dict(ds4.metadata))

    prior_gauss = NIRLiteratureGaussianPrior().compute(kept_wl_nir)
    log(f"Prior gaussiano NIR construído (p={len(kept_wl_nir)})")

    pls_nir = PLSModel(n_components=args.n_components_pls).fit(train_ds_nir)
    model_nir_gauss = PGSGv2Model(
        hidden=args.hidden, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed,
    ).fit(train_ds_nir, prior=prior_gauss)

    r2_pls_nir = r2_score(test_ds_nir.y, pls_nir.predict(test_ds_nir))
    r2_pgsg_nir_gauss = r2_score(test_ds_nir.y, model_nir_gauss.predict(test_ds_nir))
    topo_nir_gauss = gate_topology(model_nir_gauss.gates)
    rho_nir_gauss = float(np.corrcoef(model_nir_gauss.gates, prior_gauss)[0, 1])

    log(f"NIR (prior gaussiano): R2_PLS={r2_pls_nir:.4f}  R2_PGSGv2={r2_pgsg_nir_gauss:.4f}  "
        f"best_epoch={model_nir_gauss.train_history['best_epoch']}  rho={rho_nir_gauss:.4f}")
    log(f"Topologia (prior gaussiano): {topo_nir_gauss}")

    # ================= (b) Suavidade do espectro bruto =================
    log("--- Parte (b): suavidade do espectro bruto (independente de prior) ---")
    tv_spectrum_nir = spectrum_smoothness(X_tr_nir)
    log(f"NIR: suavidade do espectro bruto (pré-processado) = {tv_spectrum_nir:.6f}")

    log("Carregando e pré-processando Raman (bioprocess_substrates) para comparação ...")
    ds2 = RamanBioprocessSubstratesLoader().load()
    ds2 = RamanPreprocessor().fit_transform(ds2)
    ds1_raman = to_pgsg1_dataset(ds2, target_unit="unknown")
    idx_r = np.arange(ds1_raman.n_samples)
    train_idx_r, _ = train_test_split(idx_r, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    X_tr_raman = ds1_raman.X[train_idx_r]
    tv_spectrum_raman = spectrum_smoothness(X_tr_raman)
    log(f"Raman: suavidade do espectro bruto (pré-processado) = {tv_spectrum_raman:.6f}")

    print()
    print("=" * 70)
    print("RESUMO DA ABLAÇÃO H3")
    print("=" * 70)
    print(f"(a) Topologia do GATE com prior de mesma forma funcional (gaussiano):")
    print(f"    NIR   : entropy={topo_nir_gauss['entropy']:.4f}  smoothness={topo_nir_gauss['smoothness']:.6f}  sparsity={topo_nir_gauss['sparsity_hoyer']:.4f}")
    print(f"    (Raman de referência, prior já gaussiano: entropy=6.8549 smoothness=0.007570 sparsity=0.4150)")
    print()
    print(f"(b) Suavidade do ESPECTRO BRUTO (independente de prior/gate):")
    print(f"    NIR   : {tv_spectrum_nir:.6f}")
    print(f"    Raman : {tv_spectrum_raman:.6f}")
    print(f"    Razão NIR/Raman: {tv_spectrum_nir/tv_spectrum_raman:.3f}")
    print("=" * 70)

    np.savez(
        "ablation_h3_result.npz",
        gate_nir_gauss=model_nir_gauss.gates, prior_nir_gauss=prior_gauss,
        wavelengths_nir=kept_wl_nir,
        r2_pls_nir=r2_pls_nir, r2_pgsg_nir_gauss=r2_pgsg_nir_gauss,
        rho_nir_gauss=rho_nir_gauss,
        entropy_nir_gauss=topo_nir_gauss["entropy"],
        smoothness_nir_gauss=topo_nir_gauss["smoothness"],
        sparsity_nir_gauss=topo_nir_gauss["sparsity_hoyer"],
        tv_spectrum_nir=tv_spectrum_nir,
        tv_spectrum_raman=tv_spectrum_raman,
    )
    log("Resultado salvo em ablation_h3_result.npz")


if __name__ == "__main__":
    main()
