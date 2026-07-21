"""
Contrato do operador de ingestão (G) do pipeline PGSG, estendido para
múltiplas modalidades espectroscópicas (pgsg_2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Domain = Literal["nir", "raman"]
WavelengthUnit = Literal["nm", "cm-1"]


@dataclass
class SpectralDataset:
    """Contrato de dados que atravessa os operadores G -> T -> P -> C -> M."""

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    domain: Domain
    wavelength_unit: WavelengthUnit
    target_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y).reshape(-1)
        self.wavelengths = np.asarray(self.wavelengths).reshape(-1)

        if self.X.ndim != 2:
            raise ValueError(f"X deve ser 2D (n_amostras, n_bandas); recebido {self.X.ndim}D")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X e y têm números de amostras diferentes: "
                f"{self.X.shape[0]} vs {self.y.shape[0]}"
            )
        if self.X.shape[1] != self.wavelengths.shape[0]:
            raise ValueError(
                f"Número de bandas em X ({self.X.shape[1]}) não corresponde "
                f"ao eixo wavelengths ({self.wavelengths.shape[0]})"
            )
        if self.domain not in ("nir", "raman"):
            raise ValueError(f"domain inválido: {self.domain!r}")
        if self.wavelength_unit not in ("nm", "cm-1"):
            raise ValueError(f"wavelength_unit inválido: {self.wavelength_unit!r}")
        if np.isnan(self.X).any():
            raise ValueError("X contém NaN -- pré-processamento de ingestão incompleto")
        if np.isnan(self.y).any():
            raise ValueError("y contém NaN -- amostras sem rótulo devem ser filtradas na ingestão")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_bands(self) -> int:
        return self.X.shape[1]


class IngestionOperator(ABC):
    """Contrato do operador G: cada modalidade/base implementa seu carregador."""

    @abstractmethod
    def load(self) -> SpectralDataset:
        raise NotImplementedError
