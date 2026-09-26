"""PGSGv2Model — Prior-Guided Spectral Gating v2.

Diferenças em relação ao PGSGModel original
--------------------------------------------
1. Gates via sigmoid independente por banda (não softmax global).
   Motivação: softmax com p~280 bandas produz gradientes ~60x menores
   que sigmoid, impedindo convergência do gating (demonstrado empiricamente).

2. Regressor MLP leve (não PLS) — co-otimização end-to-end via autograd.
   Motivação: PLS não é diferenciável; recalculá-lo a cada época sem
   gradiente fluindo para theta inviabiliza o aprendizado conjunto.

3. Prior de literatura (independente dos dados de treino).
   Motivação: prior calculado nos próprios dados (VIP) cria circularidade
   — os gates convergem para o prior independentemente dos dados novos.

Arquitetura
-----------
    theta ∈ R^p              (parâmetro aprendível)
    g = sigmoid(theta)       (gates independentes em (0,1))
    X_gated = X * g          (ponderação espectral)
    h = ReLU(X_gated @ W1 + b1)   (camada oculta, hidden=32)
    y_hat = h @ W2 + b2      (saída escalar)

Inicialização de theta com prior s ∈ (0,1)^p:
    theta_0 = logit(s) = log(s/(1-s))
    → sigmoid(theta_0) = s   (gates iniciam exatamente no prior)

Otimizador: Adam, lr=1e-3, weight_decay=1e-4 (Ridge implícito no MLP).
Early stopping por val_loss (20% do treino, split estratificado em y).

Interface
---------
Segue GatedModel (base.py): _fit_impl, _predict_impl, _gates_impl,
_prior_used_impl. Compatível com evaluate() e compute_interp().

Referência prior de literatura para DMC% em mango NIR
------------------------------------------------------
Saranwong et al. (2004) Postharvest Biol. Technol. 31:253-261.
Guthrie et al. (2005) Aust. J. Agric. Res. 56:1197-1204.
Bandas relevantes: 680-700nm (clorofila), 900-1000nm (2° sobreton C-H),
1100-1200nm (1° sobreton C-H).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from pgsg_1.ingestion import SpectralDataset
from pgsg_1.models.base import GatedModel

_EPS = 1e-6
_VAL_FRAC = 0.2


def _val_split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    n_val = max(1, int(n * val_frac))
    sorted_idx = np.argsort(y)
    step = max(1, n // n_val)
    val_idx = sorted_idx[::step][:n_val]
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


class _GatedMLP(nn.Module):
    """Rede interna: sigmoid gate + MLP leve."""

    def __init__(self, p: int, hidden: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(p))
        self.mlp = nn.Sequential(
            nn.Linear(p, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.theta)
        return self.mlp(X * g)

    def gates_np(self) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.theta).numpy().copy()


def make_literature_prior(wavelengths: np.ndarray) -> np.ndarray:
    """Constrói prior de literatura para DMC% em mango NIR.

    Regiões de absorção relevantes:
    - 680-700 nm: clorofila (indicador de maturidade)
    - 900-1000 nm: 2° sobreton C-H (açúcares, DMC)
    - 1100-1200 nm: 1° sobreton C-H

    Parâmetros
    ----------
    wavelengths : np.ndarray
        Comprimentos de onda em nm (após remoção de bandas zeradas).

    Retorna
    -------
    np.ndarray em (0, 1]^p, normalizado pelo máximo.
    """
    prior = np.ones(len(wavelengths)) * 0.1
    for i, w in enumerate(wavelengths):
        if 680 <= w <= 700:
            prior[i] = 0.6
        if 900 <= w <= 1000:
            prior[i] = 1.0
        if 1100 <= w <= 1200:
            prior[i] = 0.8
    return prior / prior.max()


class PGSGv2Model(GatedModel):
    """PGSG v2: sigmoid gate + MLP end-to-end + prior de literatura.

    Parâmetros
    ----------
    hidden : int
        Unidades na camada oculta do MLP (padrão 32).
    lr : float
        Taxa de aprendizado Adam (padrão 1e-3).
    weight_decay : float
        Regularização L2 (padrão 1e-4).
    max_epochs : int
        Épocas máximas (padrão 500).
    patience : int
        Early stopping (padrão 30).
    batch_size : int
        Mini-batch (padrão 256).
    seed : int
        Reprodutibilidade (padrão 42).
    """

    def __init__(
        self,
        *,
        hidden: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 500,
        patience: int = 30,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self._net: _GatedMLP | None = None
        self._prior: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._train_history: dict | None = None

    @property
    def name(self) -> str:
        return "PGSGv2"

    def _fit_impl(
        self, train: SpectralDataset, prior: np.ndarray | None
    ) -> None:
        torch.manual_seed(self.seed)
        X = train.X.astype(np.float32)
        y = train.y.astype(np.float32)
        p = X.shape[1]

        # normalizar y internamente
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y_norm = (y - self._y_mean) / self._y_std

        self._prior = prior.copy() if prior is not None else None

        # rede
        net = _GatedMLP(p, self.hidden)

        # inicializar theta com prior via logit
        if prior is not None:
            s = np.clip(prior.astype(np.float64), _EPS, 1 - _EPS)
            theta_init = torch.tensor(
                np.log(s / (1 - s)), dtype=torch.float32
            )
            with torch.no_grad():
                net.theta.copy_(theta_init)

        # split treino/val
        X_tr, y_tr, X_val, y_val = _val_split(X, y_norm, _VAL_FRAC, self.seed)
        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val).unsqueeze(1)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = 0
        no_improve = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        n_tr = len(X_tr)

        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(n_tr)
            epoch_loss = 0.0
            for i in range(0, n_tr, self.batch_size):
                b = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(net(X_tr_t[b]), y_tr_t[b])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(b)
            train_losses.append(epoch_loss / n_tr)

            net.eval()
            with torch.no_grad():
                val_loss = criterion(net(X_val_t), y_val_t).item()
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        net.load_state_dict(best_state)
        self._net = net
        self._train_history = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def _predict_impl(self, dataset: SpectralDataset) -> np.ndarray:
        X = torch.from_numpy(dataset.X.astype(np.float32))
        self._net.eval()
        with torch.no_grad():
            y_norm = self._net(X).numpy().ravel()
        return y_norm * self._y_std + self._y_mean

    def _gates_impl(self) -> np.ndarray:
        return self._net.gates_np()

    def _prior_used_impl(self) -> np.ndarray | None:
        return self._prior

    @property
    def train_history(self) -> dict | None:
        return self._train_history
