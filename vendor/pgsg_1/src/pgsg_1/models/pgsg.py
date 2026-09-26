"""PGSGModel — Prior-Guided Spectral Gating para regressão espectral.

Arquitetura
-----------
O PGSG aprende um vetor de gates g ∈ [0,1]^p que pondera as bandas
antes de um regressor linear (PLS-1). A hipótese central do projeto é
que inicializar theta com um prior s derivado de literatura acelera a
convergência e melhora a interpretabilidade em regimes de poucos dados.

Formulação
----------
    theta ∈ R^p          parâmetros do gating (não normalizados)
    g = softmax(theta)   gates em [0,1]^p, soma 1
    X_gated = X * g      multiplicação elemento-a-elemento (broadcasting)
    y_hat = PLS(X_gated) regressão PLS-1 sobre o espectro ponderado

Inicialização de theta
----------------------
    Com prior s ∈ [0,1]^p:
        theta_0 = log(s + eps) - mean(log(s + eps))
        (transformação logit-centrada; garante softmax(theta_0) ≈ s)
    Sem prior (s=None):
        theta_0 = zeros(p)  → softmax = uniforme 1/p

Treino de theta
---------------
Gradiente descendente sobre MSE(y, PLS(X * softmax(theta))):
    - PLS é reajustado a cada época com X * softmax(theta) atual
    - Gradiente de theta estimado por diferença finita (central differences)
      com step h=1e-3; evita dependência de autograd e é estável para p~265
    - Otimizador: SGD com momentum (sem Adam, para reproduzir o paper)
    - Early stopping por val_loss (20% do treino, split por quantis de y)
    - Melhor theta restaurado ao final

Referência
----------
Artigo fundador PGSG (2023), Seção 3.2: "Gate Learning via Prior
Initialization". A implementação segue o pseudocódigo da Figura 2.

Interface
---------
Segue GatedModel (base.py):
    _fit_impl, _predict_impl, _gates_impl, _prior_used_impl
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from pgsg_1.ingestion import SpectralDataset

from .base import GatedModel

# ---------------------------------------------------------------------------
# constantes
# ---------------------------------------------------------------------------

_EPS = 1e-8          # estabilidade numérica no log
_H = 1e-3            # step para diferença finita
_VAL_FRAC = 0.2      # fração de validação interna
_MOMENTUM = 0.9      # momentum do SGD


# ---------------------------------------------------------------------------
# helpers internos
# ---------------------------------------------------------------------------

def _softmax(theta: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável."""
    e = np.exp(theta - theta.max())
    return e / e.sum()


def _pls_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, PLSRegression]:
    """Ajusta PLS em (X_tr, y_tr) e prediz X_te. Retorna (y_hat, pls)."""
    n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_tr, y_tr)
    return pls.predict(X_te).ravel(), pls


def _val_split(
    X: np.ndarray, y: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split treino/val estratificado por quantis de y."""
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


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# PGSGModel
# ---------------------------------------------------------------------------

class PGSGModel(GatedModel):
    """Prior-Guided Spectral Gating para regressão espectral.

    Parâmetros
    ----------
    n_components : int
        Número de componentes do PLS interno (padrão 10).
    lr : float
        Taxa de aprendizado do SGD (padrão 1e-2).
    max_epochs : int
        Épocas máximas de treino (padrão 200).
    patience : int
        Early stopping: épocas sem melhora na val loss (padrão 20).
    h : float
        Step para diferença finita na estimação do gradiente (padrão 1e-3).
    momentum : float
        Momentum do SGD (padrão 0.9).
    seed : int
        Seed para reprodutibilidade do split de validação (padrão 42).
    """

    def __init__(
        self,
        *,
        n_components: int = 10,
        lr: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 20,
        h: float = _H,
        momentum: float = _MOMENTUM,
        seed: int = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components deve ser >= 1, recebido {n_components}")
        if lr <= 0:
            raise ValueError(f"lr deve ser > 0, recebido {lr}")
        super().__init__()
        self.n_components = n_components
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.h = h
        self.momentum = momentum
        self.seed = seed

        self._theta: np.ndarray | None = None
        self._pls: PLSRegression | None = None
        self._prior: np.ndarray | None = None
        self._train_history: dict | None = None

    # ------------------------------------------------------------------
    # Interface GatedModel
    # ------------------------------------------------------------------

    def _fit_impl(
        self, train: SpectralDataset, prior: np.ndarray | None
    ) -> None:
        X = train.X.astype(np.float64)
        y = train.y.astype(np.float64)
        p = X.shape[1]

        # inicialização de theta
        self._prior = prior.copy() if prior is not None else None
        if prior is not None:
            s = np.clip(prior.astype(np.float64), _EPS, 1.0)
            log_s = np.log(s)
            theta = log_s - log_s.mean()
        else:
            theta = np.zeros(p)

        # split treino / validação interna
        X_tr, y_tr, X_val, y_val = _val_split(X, y, _VAL_FRAC, self.seed)

        best_val_loss = float("inf")
        best_theta = theta.copy()
        best_pls = None
        epochs_no_improve = 0
        velocity = np.zeros(p)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = 0

        for epoch in range(self.max_epochs):
            g = _softmax(theta)

            # PLS no treino ponderado
            y_hat_tr, pls = _pls_predict(X_tr * g, y_tr, X_tr * g, self.n_components)
            train_losses.append(_mse(y_tr, y_hat_tr))

            # val loss
            y_hat_val = pls.predict(X_val * g).ravel()
            val_loss = _mse(y_val, y_hat_val)
            val_losses.append(val_loss)

            # gradiente por diferença finita central
            grad = np.zeros(p)
            for j in range(p):
                theta_p = theta.copy(); theta_p[j] += self.h
                theta_m = theta.copy(); theta_m[j] -= self.h
                g_p = _softmax(theta_p)
                g_m = _softmax(theta_m)
                y_p = pls.predict(X_tr * g_p).ravel()
                y_m = pls.predict(X_tr * g_m).ravel()
                grad[j] = (_mse(y_tr, y_p) - _mse(y_tr, y_m)) / (2 * self.h)

            # SGD com momentum
            velocity = self.momentum * velocity + self.lr * grad
            theta = theta - velocity

            # early stopping
            if val_loss < best_val_loss - _EPS:
                best_val_loss = val_loss
                best_theta = theta.copy()
                best_pls = copy.deepcopy(pls)
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        # restaurar melhor estado
        self._theta = best_theta
        # reajustar PLS com theta final em TODO o treino (não só X_tr)
        g_best = _softmax(best_theta)
        _, self._pls = _pls_predict(X * g_best, y, X * g_best, self.n_components)

        self._train_history = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def _predict_impl(self, dataset: SpectralDataset) -> np.ndarray:
        X = dataset.X.astype(np.float64)
        g = _softmax(self._theta)
        return self._pls.predict(X * g).ravel()

    def _gates_impl(self) -> np.ndarray:
        return _softmax(self._theta)

    def _prior_used_impl(self) -> np.ndarray | None:
        return self._prior

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    @property
    def train_history(self) -> dict | None:
        return self._train_history
