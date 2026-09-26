"""CNN1DModel."""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import numpy as np
import torch
import torch.nn as nn
from .base import Model
from .training import train_torch_model
if TYPE_CHECKING:
    from pgsg_1.ingestion import SpectralDataset

class _CNNShallow(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,kernel_size=5,padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32*16,64), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, x): return self.head(self.conv(x.unsqueeze(1)))

class _CNNDeep(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,kernel_size=5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,kernel_size=3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128*8,128), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(128,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.head(self.conv(x.unsqueeze(1)))

class CNN1DModel(Model):
    def __init__(self, *, depth="shallow", dropout=0.1, lr=1e-3, max_epochs=200,
                 patience=20, batch_size=32, seed=42):
        if depth not in ("shallow","deep"): raise ValueError(f"depth deve ser 'shallow' ou 'deep', recebido '{depth}'")
        if not 0.0 <= dropout < 1.0: raise ValueError(f"dropout deve estar em [0, 1), recebido {dropout}")
        super().__init__()
        self.depth=depth; self.dropout=dropout; self.lr=lr
        self.max_epochs=max_epochs; self.patience=patience
        self.batch_size=batch_size; self.seed=seed
        self._net=None; self._train_history=None

    @property
    def name(self): return f"CNN1D-{self.depth}"

    def _fit_impl(self, train, prior):
        torch.manual_seed(self.seed)
        self._net = _CNNShallow(self.dropout) if self.depth=="shallow" else _CNNDeep(self.dropout)
        self._train_history = train_torch_model(
            self._net, train, lr=self.lr, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size, seed=self.seed)

    def _predict_impl(self, dataset):
        X = torch.from_numpy(np.asarray(dataset.X, dtype=np.float32))
        self._net.eval()
        with torch.no_grad(): return self._net(X).numpy().ravel()

    @property
    def train_history(self): return self._train_history
