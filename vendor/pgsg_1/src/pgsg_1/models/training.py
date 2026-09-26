"""Utilitários de treino PyTorch para o operador M.

Fornece um loop de treino genérico com early stopping baseado em loss de
validação. Usado por MLPModel e CNN1DModel; será reutilizado por PGSGModel
na iteração 2.

Design
------
- `train_torch_model` é a única função pública; os modelos passam o net,
  os dados já convertidos em tensores, e os hiperparâmetros.
- Early stopping monitora a loss de validação (MSE); restaura os pesos
  do melhor epoch ao final.
- Validação interna: 20% do treino (split estratificado por valor de y,
  via quantis). A divisão usa seed fixa para reprodutibilidade.
- Retorna dict com histórico para inspeção (útil nos testes).

Invariantes
-----------
- O modelo recebido é modificado in-place (pesos do melhor epoch).
- A seed do torch é restaurada ao valor anterior após o treino.
- Nunca toca nos dados de teste — recebe apenas train_dataset.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from pgsg_1.ingestion import SpectralDataset


def train_torch_model(
    net: nn.Module,
    train_dataset: "SpectralDataset",
    *,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Treina *net* em train_dataset com early stopping.

    Parâmetros
    ----------
    net:
        Módulo PyTorch a treinar (modificado in-place).
    train_dataset:
        SpectralDataset com atributos .X (n×p) e .y (n,).
    lr:
        Taxa de aprendizado (Adam).
    max_epochs:
        Número máximo de épocas.
    patience:
        Épocas sem melhora antes de parar.
    batch_size:
        Tamanho do mini-batch.
    val_fraction:
        Fração do treino usada como validação interna.
    seed:
        Seed para reprodutibilidade do split e do DataLoader.

    Retorna
    -------
    dict com chaves:
        - 'best_epoch': int
        - 'best_val_loss': float
        - 'train_losses': list[float]
        - 'val_losses': list[float]
    """
    # --- salva estado do RNG para restaurar depois -------------------------
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    try:
        X = np.asarray(train_dataset.X, dtype=np.float32)
        y = np.asarray(train_dataset.y, dtype=np.float32).reshape(-1, 1)
        n = len(X)

        # --- split treino / validação interna --------------------------------
        n_val = max(1, int(n * val_fraction))
        # split por quantis de y para evitar desbalanceamento
        sorted_idx = np.argsort(y.ravel())
        val_idx = sorted_idx[::max(1, n // n_val)][:n_val]
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # --- DataLoaders -----------------------------------------------------
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))

        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        # --- otimizador e critério -------------------------------------------
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(net.state_dict())
        best_epoch = 0
        epochs_without_improvement = 0

        train_losses: list[float] = []
        val_losses: list[float] = []

        # --- loop principal --------------------------------------------------
        for epoch in range(max_epochs):
            net.train()
            epoch_loss = 0.0
            for Xb, yb in dl_tr:
                optimizer.zero_grad()
                pred = net(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)
            train_losses.append(epoch_loss / len(X_tr))

            # validação
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        # restaura melhores pesos
        net.load_state_dict(best_weights)

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    finally:
        torch.set_rng_state(rng_state)
