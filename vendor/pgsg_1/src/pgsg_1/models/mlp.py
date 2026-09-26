"""MLPModel."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
from .base import Model
from .training import train_torch_model
if TYPE_CHECKING:
    from pgsg_1.ingestion import SpectralDataset

class _MLP(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x): return self.net(x)

class MLPModel(Model):
    def __init__(self, *, hidden=64, dropout=0.1, lr=1e-3, max_epochs=200,
                 patience=20, batch_size=32, seed=42):
        if hidden < 2: raise ValueError(f"hidden deve ser >= 2, recebido {hidden}")
        if not 0.0 <= dropout < 1.0: raise ValueError(f"dropout deve estar em [0, 1), recebido {dropout}")
        super().__init__()
        self.hidden=hidden; self.dropout=dropout; self.lr=lr
        self.max_epochs=max_epochs; self.patience=patience
        self.batch_size=batch_size; self.seed=seed
        self._net=None; self._train_history=None

    def _fit_impl(self, train, prior):
        torch.manual_seed(self.seed)
        self._net = _MLP(train.X.shape[1], self.hidden, self.dropout)
        self._train_history = train_torch_model(
            self._net, train, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size, seed=self.seed)

    def _predict_impl(self, dataset):
        X = torch.from_numpy(np.asarray(dataset.X, dtype=np.float32))
        self._net.eval()
        with torch.no_grad(): return self._net(X).numpy().ravel()

    @property
    def train_history(self): return self._train_history
