"""Operador M — modelagem."""
from .base import GatedModel, Model
from .cnn import CNN1DModel
from .mlp import MLPModel
from .pgsg import PGSGModel
from .pls import PLSModel

__all__ = ["Model", "GatedModel", "PLSModel", "MLPModel", "CNN1DModel", "PGSGModel"]
