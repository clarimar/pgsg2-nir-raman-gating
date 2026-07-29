"""
Script diagnóstico para investigar best_epoch=0 observado na rodada
completa de run_h1_raman.py (H1 não suportada, Delta_R2_PLS ~ 0).

Mantém p=1870 (bandas) completo -- é a dimensão sob suspeita de
gradiente vanescente do softmax em alta dimensão -- e reduz apenas n
(amostras) para tornar a iteração mais rápida.

Salva train_history (train_losses, val_losses por época) em .npz para
inspeção, em vez de só reportar best_epoch como o script principal.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.model_selection import train_test_split

from pgsg2.ingestion.raman_bioprocess import RamanBioprocessSubstratesLoader
from pgsg2.preprocessing.raman import RamanPreprocessor
from pgsg2.priors.raman import RamanGlucosePrior
from pgsg2.models.adapter import to_pgsg1_dataset

from pgsg_1.ingestion import SpectralDataset as PGSG1SpectralDataset
from pgsg_1.models.pgsg import PGSGModel


def _subsample_samples_only(ds1: PGSG1SpectralDataset, n_samples: int, seed: int) -> PGSG1SpectralDataset:
    """Subamostra só amostras, preservando todas as p=1870 bandas."""
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, ds1.n_samples)
    idx = rng.choice(ds1.n_samples, size=n_samples, replace=False)
    idx.sort()
    return PGSG1SpectralDataset(
        X=ds1.X[idx], y=ds1.y[idx], wavelengths=ds1.wavelengths,
        metadata=dict(ds1.metadata),
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico: trajetória de val_loss do PGSGModel em Raman (p completo)")
    parser.add_argument("--n-samples", type=int, default=1000, help="amostras a manter (bandas ficam completas, p=1870)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)  # sem early stop, para ver a curva inteira
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="diagnostic_history.npz")
    args = parser.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    log("Carregando bioprocess_substrates ...")
    ds2 = RamanBioprocessSubstratesLoader().load()
    log(f"Carregado: n={ds2.n_samples}, p={ds2.n_bands}")

    log("Pré-processando ...")
    ds2 = RamanPreprocessor().fit_transform(ds2)
    log("Pré-processamento concluído")

    ds1 = to_pgsg1_dataset(ds2, target_unit="unknown")
    ds1 = _subsample_samples_only(ds1, n_samples=args.n_samples, seed=args.seed)
    log(f"Subamostrado (bandas completas): n={ds1.n_samples}, p={ds1.n_bands}")

    idx = np.arange(ds1.n_samples)
    train_idx, test_idx = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    train = PGSG1SpectralDataset(X=ds1.X[train_idx], y=ds1.y[train_idx], wavelengths=ds1.wavelengths, metadata=dict(ds1.metadata))

    prior = RamanGlucosePrior().compute(train.wavelengths)

    log(f"Treinando PGSGModel diagnóstico (max_epochs={args.max_epochs}, patience={args.patience}, lr={args.lr}) ...")
    t_fit0 = time.time()
    model = PGSGModel(
        n_components=args.n_components, max_epochs=args.max_epochs,
        patience=args.patience, lr=args.lr, seed=args.seed,
    ).fit(train, prior=prior)
    log(f"Treino concluído em {time.time()-t_fit0:.1f}s (best_epoch={model.train_history['best_epoch']})")

    hist = model.train_history
    train_losses = np.array(hist["train_losses"])
    val_losses = np.array(hist["val_losses"])

    np.savez(
        args.out,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=hist["best_epoch"],
        best_val_loss=hist["best_val_loss"],
        n_samples=ds1.n_samples,
        n_bands=ds1.n_bands,
    )
    log(f"Histórico salvo em {args.out}")

    print()
    print("Época | train_loss | val_loss")
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        marker = " <-- best" if i == hist["best_epoch"] else ""
        print(f"{i:5d} | {tl:10.4f} | {vl:10.4f}{marker}")

    print()
    print(f"val_loss[0]  = {val_losses[0]:.4f}")
    print(f"val_loss[-1] = {val_losses[-1]:.4f}")
    print(f"val_loss min = {val_losses.min():.4f} (época {val_losses.argmin()})")
    print(f"val_loss monotonicamente crescente após época 0? "
          f"{bool(np.all(np.diff(val_losses[0:]) >= -1e-9))}")


if __name__ == "__main__":
    main()
