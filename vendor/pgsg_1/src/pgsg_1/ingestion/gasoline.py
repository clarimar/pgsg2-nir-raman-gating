"""Loader do dataset Gasoline (octanagem por NIR).

Formato esperado do CSV
-----------------------
Arquivo derivado do dataset `gasoline` do pacote R `pls` (Mevik & Wehrens 2007).
Distribuído publicamente em https://github.com/gustavovelascoh/octane-NIR
como `gasoline.csv`.

Estrutura: 60 linhas × 402 colunas
    - coluna 0:   'octane'                       — alvo
    - colunas 1..401: 'NIR.900 nm' .. 'NIR.1700 nm' — absorbância log(1/R)

Os comprimentos de onda variam de 900 a 1700 nm em passos de 2 nm
(401 bandas igualmente espaçadas). Espectros adquiridos por refletância
difusa.

Sobre a faixa espectral
-----------------------
A faixa 900–1700 nm é SWIR (short-wave infrared), distinta da faixa do
Tecator (850–1050 nm). As regiões de absorção química relevantes são
diferentes: para gasolina, sobretons de C–H em ~1200 nm e ~1400 nm.

Referências
-----------
- Mevik, B. H.; Wehrens, R. (2007). The pls package: principal component
  and partial least squares regression in R. Journal of Statistical
  Software, 18(2), 1–23.
- Kalivas, J. H. (cortesia dos dados originais).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import SpectralDataset

# constantes da base
_N_SAMPLES: int = 60
_N_BANDS: int = 401
_WL_START_NM: float = 900.0
_WL_END_NM: float = 1700.0
_WL_STEP_NM: float = 2.0
_TARGET_COL: str = "octane"

# regex para extrair o numero do nome da coluna 'NIR.900 nm', 'NIR.902 nm', ...
_WL_PATTERN = re.compile(r"(\d+)")


def load_gasoline(csv_path: str | Path) -> SpectralDataset:
    """Carrega Gasoline a partir do CSV exportado do pacote R `pls`.

    Parâmetros
    ----------
    csv_path : str | Path
        Caminho para `gasoline.csv`.

    Retorna
    -------
    SpectralDataset
        Tripla canônica (X, y, wavelengths) com metadados.

    Erros
    -----
    FileNotFoundError
        Se `csv_path` não existir.
    ValueError
        Se o CSV não tiver o shape esperado, faltar a coluna 'octane', ou
        a quantidade/passo das bandas espectrais não bater com a
        especificação (900–1700 nm em passos de 2 nm).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo do Gasoline não encontrado: {csv_path}. "
            f"Use `scripts/download_gasoline.py` para baixá-lo."
        )

    df = pd.read_csv(csv_path)

    # shape
    if df.shape[0] != _N_SAMPLES:
        raise ValueError(
            f"Gasoline: esperado {_N_SAMPLES} linhas, encontrado {df.shape[0]}"
        )
    n_cols_esperado = 1 + _N_BANDS  # octane + 401 bandas
    if df.shape[1] != n_cols_esperado:
        raise ValueError(
            f"Gasoline: esperado {n_cols_esperado} colunas "
            f"(1 alvo + {_N_BANDS} bandas), encontrado {df.shape[1]}"
        )

    # alvo
    if _TARGET_COL not in df.columns:
        raise ValueError(
            f"Gasoline: coluna alvo {_TARGET_COL!r} ausente; "
            f"colunas presentes: {list(df.columns[:3])}..."
        )
    y = df[_TARGET_COL].to_numpy(dtype=np.float64)

    # bandas: todas as colunas que não são 'octane', mantendo a ordem do arquivo
    band_cols = [c for c in df.columns if c != _TARGET_COL]
    X = df[band_cols].to_numpy(dtype=np.float64)

    # extrai comprimentos de onda dos nomes das colunas
    wavelengths_from_names = _parse_wavelengths(band_cols)

    # verifica passo regular de 2 nm e faixa correta
    _validate_wavelength_grid(wavelengths_from_names)

    metadata = {
        "domain": "gasoline",
        "target_name": "octane",
        "target_unit": "octane number",
        "wavelength_unit": "nm",
        "source": "https://github.com/gustavovelascoh/octane-NIR/blob/master/gasoline.csv",
        "original_source": "R package 'pls' (Mevik & Wehrens, 2007); data from J. H. Kalivas",
        "wavelength_range_nm": (_WL_START_NM, _WL_END_NM),
        "wavelength_step_nm": _WL_STEP_NM,
        "spectrum_type": "diffuse reflectance log(1/R)",
    }

    return SpectralDataset(
        X=X,
        y=y,
        wavelengths=wavelengths_from_names,
        metadata=metadata,
    )


# helpers --------------------------------------------------------------------


def _parse_wavelengths(col_names: list[str]) -> np.ndarray:
    """Extrai comprimentos de onda dos nomes 'NIR.900 nm', 'NIR.902 nm', ...

    Retorna array float64 de tamanho len(col_names).
    """
    out = np.empty(len(col_names), dtype=np.float64)
    for i, name in enumerate(col_names):
        m = _WL_PATTERN.search(name)
        if m is None:
            raise ValueError(
                f"Não consegui extrair comprimento de onda da coluna {name!r}. "
                f"Esperado formato como 'NIR.900 nm'."
            )
        out[i] = float(m.group(1))
    return out


def _validate_wavelength_grid(wavelengths: np.ndarray) -> None:
    """Valida que a grade é 900–1700 nm com passo de 2 nm."""
    if wavelengths[0] != _WL_START_NM:
        raise ValueError(
            f"Gasoline: primeira banda esperada {_WL_START_NM} nm, "
            f"encontrada {wavelengths[0]} nm"
        )
    if wavelengths[-1] != _WL_END_NM:
        raise ValueError(
            f"Gasoline: última banda esperada {_WL_END_NM} nm, "
            f"encontrada {wavelengths[-1]} nm"
        )
    diffs = np.diff(wavelengths)
    if not np.allclose(diffs, _WL_STEP_NM):
        raise ValueError(
            f"Gasoline: passo de banda inconsistente; esperado {_WL_STEP_NM} nm. "
            f"Min={diffs.min()}, max={diffs.max()}"
        )
