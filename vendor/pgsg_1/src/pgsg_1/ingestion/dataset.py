"""Estrutura de dados canônica: tripla (X, y, lambda) com metadados.

Esta é a saída canônica do operador G (ingestion) e a entrada de todos os
estágios subsequentes. Qualquer base — Tecator, Gasoline, Mango DMC v3 — é
absorvida no mesmo contrato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectralDataset:
    """Tripla canônica de uma base espectral.

    Atributos
    ---------
    X : np.ndarray, shape (n, p)
        Matriz espectral. Linhas = amostras, colunas = bandas.
    y : np.ndarray, shape (n,)
        Vetor de alvos contínuos.
    wavelengths : np.ndarray, shape (p,)
        Comprimentos de onda em nm, alinhados às colunas de X.
    metadata : dict
        Metadados essenciais (domínio, unidade do alvo, faixa espectral,
        identificadores de grupo quando aplicável). Ver lista esperada em
        `REQUIRED_METADATA_KEYS`.
    sample_ids : np.ndarray | None, shape (n,)
        Identificadores opcionais por amostra (útil para rastrear safras,
        cultivares, instrumentos na Mango DMC v3).
    group_ids : np.ndarray | None, shape (n,)
        Identificadores opcionais de grupo para estratificação (safra ou
        outro agrupamento natural). Usado pelo operador C (scenarios).

    Invariantes (Princípio da Reprodutibilidade Determinística)
    -----------------------------------------------------------
    I1. X.shape == (n, p) e y.shape == (n,) e wavelengths.shape == (p,)
    I2. wavelengths é estritamente crescente
    I3. nenhum NaN em X, y, wavelengths
    I4. metadata contém todas as chaves de REQUIRED_METADATA_KEYS

    `frozen=True` torna a instância imutável: bases não podem ser
    alteradas em silêncio depois de carregadas.
    """

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_ids: np.ndarray | None = None
    group_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        # validações dos invariantes I1–I4
        _validate(self)

    # propriedades convenientes ----------------------------------------------

    @property
    def n_samples(self) -> int:
        """Número de amostras (n)."""
        return self.X.shape[0]

    @property
    def n_bands(self) -> int:
        """Número de bandas espectrais (p)."""
        return self.X.shape[1]

    @property
    def domain(self) -> str:
        """Identificador de domínio (vem dos metadados)."""
        return self.metadata.get("domain", "unknown")

    def __repr__(self) -> str:
        return (
            f"SpectralDataset(domain={self.domain!r}, "
            f"n={self.n_samples}, p={self.n_bands}, "
            f"wavelengths=[{self.wavelengths[0]:.1f}, {self.wavelengths[-1]:.1f}] nm)"
        )


# chaves obrigatórias em metadata (definem o contrato mínimo)
REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "domain",  # identificador do domínio (e.g. "tecator", "mango_dmc_v3")
        "target_name",  # nome do alvo (e.g. "fat_pct", "dmc_pct")
        "target_unit",  # unidade (e.g. "%")
        "wavelength_unit",  # unidade dos comprimentos de onda (e.g. "nm")
        "source",  # proveniência (URL, DOI, ou referência)
    }
)


def _validate(ds: SpectralDataset) -> None:
    """Valida os invariantes I1–I4 do contrato."""
    # I1: shapes consistentes
    if ds.X.ndim != 2:
        raise ValueError(f"X deve ser 2D, recebido {ds.X.ndim}D")
    n, p = ds.X.shape
    if ds.y.shape != (n,):
        raise ValueError(f"y.shape esperado ({n},), recebido {ds.y.shape}")
    if ds.wavelengths.shape != (p,):
        raise ValueError(
            f"wavelengths.shape esperado ({p},), recebido {ds.wavelengths.shape}"
        )

    # I2: wavelengths estritamente crescente
    if not np.all(np.diff(ds.wavelengths) > 0):
        raise ValueError("wavelengths deve ser estritamente crescente")

    # I3: sem NaN
    if np.isnan(ds.X).any():
        raise ValueError("X contém NaN")
    if np.isnan(ds.y).any():
        raise ValueError("y contém NaN")
    if np.isnan(ds.wavelengths).any():
        raise ValueError("wavelengths contém NaN")

    # I4: metadata mínimo
    faltando = REQUIRED_METADATA_KEYS - set(ds.metadata.keys())
    if faltando:
        raise ValueError(
            f"metadata faltando chaves obrigatórias: {sorted(faltando)}"
        )

    # sample_ids/group_ids: se presentes, devem ter o tamanho certo
    if ds.sample_ids is not None and ds.sample_ids.shape != (n,):
        raise ValueError(
            f"sample_ids.shape esperado ({n},), recebido {ds.sample_ids.shape}"
        )
    if ds.group_ids is not None and ds.group_ids.shape != (n,):
        raise ValueError(
            f"group_ids.shape esperado ({n},), recebido {ds.group_ids.shape}"
        )
