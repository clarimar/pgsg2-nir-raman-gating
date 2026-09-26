"""Loader do dataset Tecator (carne, gordura).

A base Tecator existe em três versões em circulação. Esta implementação
suporta a versão **OpenML (ID 505)**, que é a mais acessível via Python.

Estrutura do CSV OpenML
-----------------------
240 linhas (1 por amostra). Colunas, na ordem:
- 100 absorbâncias (bandas em 850–1050 nm, igualmente espaçadas)
- 22 componentes principais (descartados aqui — usamos as bandas cruas)
- 3 alvos: moisture, fat, protein

Convenção do artigo fundador
----------------------------
O artigo usa apenas as **primeiras 215 amostras** (descartando as 25
últimas, que são casos de extrapolação). O alvo padrão é `fat` (%).

Faixa espectral
---------------
850 a 1050 nm, 100 bandas igualmente espaçadas. Os comprimentos de onda
não estão no CSV — são reconstruídos via np.linspace.

Referências
-----------
- StatLib: http://lib.stat.cmu.edu/datasets/tecator
- OpenML: https://www.openml.org/d/505
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .dataset import SpectralDataset

# constantes da base
_N_BANDS: int = 100
_WL_START_NM: float = 850.0
_WL_END_NM: float = 1050.0
_N_PCS: int = 22  # componentes principais armazenados no CSV (descartados)
_TARGETS: tuple[str, str, str] = ("moisture", "fat", "protein")
_N_SAMPLES_FULL: int = 240
_N_SAMPLES_ARTIGO: int = 215  # convenção do artigo fundador (sem extrapolação)


def load_tecator(
    csv_path: str | Path,
    target: Literal["fat", "moisture", "protein"] = "fat",
    use_artigo_subset: bool = True,
) -> SpectralDataset:
    """Carrega Tecator a partir do CSV no formato OpenML.

    Parâmetros
    ----------
    csv_path : str | Path
        Caminho para `tecator.csv` (formato OpenML — ver scripts/download_tecator.py).
    target : {'fat', 'moisture', 'protein'}
        Qual alvo usar. Padrão: 'fat', alinhado com o artigo fundador.
    use_artigo_subset : bool
        Se True (padrão), usa apenas as primeiras 215 amostras, seguindo a
        convenção do artigo fundador. Se False, retorna todas as 240.

    Retorna
    -------
    SpectralDataset
        Tripla canônica (X, y, wavelengths) com metadados.

    Erros
    -----
    FileNotFoundError
        Se `csv_path` não existir.
    ValueError
        Se o CSV não tiver o número esperado de linhas/colunas, ou se o
        target solicitado for inválido.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo do Tecator não encontrado: {csv_path}. "
            f"Use `scripts/download_tecator.py` para baixá-lo."
        )

    if target not in _TARGETS:
        raise ValueError(f"target deve ser um de {_TARGETS}, recebido {target!r}")

    # lê o CSV. Formato OpenML: 125 colunas, sem índice.
    df = pd.read_csv(csv_path)

    n_esperado = _N_SAMPLES_FULL
    n_cols_esperado = _N_BANDS + _N_PCS + len(_TARGETS)
    if df.shape[0] != n_esperado:
        raise ValueError(
            f"Tecator: esperado {n_esperado} linhas, encontrado {df.shape[0]}"
        )
    if df.shape[1] != n_cols_esperado:
        raise ValueError(
            f"Tecator: esperado {n_cols_esperado} colunas "
            f"({_N_BANDS} bandas + {_N_PCS} PCs + {len(_TARGETS)} alvos), "
            f"encontrado {df.shape[1]}"
        )

    # extrai as 100 absorbâncias (primeiras 100 colunas)
    X_full = df.iloc[:, :_N_BANDS].to_numpy(dtype=np.float64)

    # extrai alvo. No formato OpenML, os 3 alvos são as 3 últimas colunas,
    # nomeadas como 'water'/'moisture', 'fat', 'protein' — usamos posição.
    target_idx = _TARGETS.index(target)
    y_full = df.iloc[:, -(len(_TARGETS) - target_idx)].to_numpy(dtype=np.float64)

    # subset do artigo fundador
    if use_artigo_subset:
        X = X_full[:_N_SAMPLES_ARTIGO]
        y = y_full[:_N_SAMPLES_ARTIGO]
    else:
        X = X_full
        y = y_full

    # comprimentos de onda reconstruídos (100 pontos igualmente espaçados)
    wavelengths = np.linspace(_WL_START_NM, _WL_END_NM, _N_BANDS, dtype=np.float64)

    metadata = {
        "domain": "tecator",
        "target_name": f"{target}_pct",
        "target_unit": "%",
        "wavelength_unit": "nm",
        "source": "https://www.openml.org/d/505",
        "n_samples_original": _N_SAMPLES_FULL,
        "n_samples_artigo_subset": _N_SAMPLES_ARTIGO,
        "uses_artigo_subset": use_artigo_subset,
        "instrument": "Tecator Infratec Food and Feed Analyzer (NIT)",
        "wavelength_range_nm": (_WL_START_NM, _WL_END_NM),
    }

    return SpectralDataset(
        X=X,
        y=y,
        wavelengths=wavelengths,
        metadata=metadata,
    )
