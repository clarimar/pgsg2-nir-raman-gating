"""Loader do dataset Mango DMC v3 (manga, teor de matéria seca).

Esta é a base de grande porte do projeto — espinha dorsal da curva de
escalabilidade. Ver ADR-0002 (escolha da base) e ADR-0004 (papel das
safras 4 e 5 como testes invariantes).

Formato esperado do CSV
-----------------------
Arquivo `MangoDMC_NIR_Data_v3.csv` do Mendeley Data
(DOI 10.17632/46htwnp833.3 ou versão 4 que mantém o mesmo arquivo).

Estrutura: 12.011 linhas × 315 colunas
    - 9 colunas de metadados:
        Set       string  'Cal' / 'Tuning' / 'Val Ext' / 'Val Ext 2'
        Season    int     1, 2, 3, 4, 5  (safra)
        Region    string  'NT', 'QLD', etc.
        Date      string  data de colheita (formato dd/mm/yyyy)
        Type      string  estágio de maturação (e.g. 'Hard Green')
        Cultivar  string  'Caly', 'KP', 'R2E2', '1201', '4069', etc.
        Pop       int     identificador de subpopulação/lote
        Temp      string  'Mid', etc.
        DM        float   dry matter content (%) — ALVO
    - 306 colunas espectrais: '285', '288', '291', ..., '1200'
      (passo de 3 nm, faixa 285–1200 nm)

Bandas sistematicamente zeradas
-------------------------------
Inspeção do arquivo real mostra que as primeiras bandas espectrais
(comprimentos de onda muito baixos, fora da faixa útil do instrumento)
contêm o valor 0 em todas as amostras. Isso NÃO é NaN, é um marcador
explícito de "sem dado válido" colocado pelos próprios autores. O loader
preserva esses zeros (X fiel ao arquivo) e registra os comprimentos de
onda afetados em `metadata['bands_systematically_zero']`, para que o
operador T (pré-processamento) saiba descartá-los.

Convenção de divisão por safra
------------------------------
Ver ADR-0004. Resumo: Season 4 (n=1448) é o teste invariante; Season 5
(n=320) é teste de generalização adicional; Seasons 1-3 (n=10.243) são
o pool de treino para a curva de escala.

Referências
-----------
- Anderson, N. T.; Walsh, K. B.; Subedi, P. P.; Hayes, C. H. (2020).
  Achieving robustness across season, location and cultivar for a NIRS
  model for intact mango fruit dry matter content. Postharvest Biology
  and Technology, 168, 111202.
- Walsh, J. et al. (2024). Evaluation of 1D Convolutional Neural Network
  in Estimation of Mango Dry Matter Content. Spectrochimica Acta Part A,
  311, 124003.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dataset import SpectralDataset

# constantes da base (validadas por inspeção do arquivo real) ----------------
_N_SAMPLES_TOTAL: int = 12_011
_N_BANDS: int = 306
_N_METADATA_COLS: int = 9
_N_TOTAL_COLS: int = _N_METADATA_COLS + _N_BANDS  # 315

_WL_START_NM: float = 285.0
_WL_END_NM: float = 1200.0
_WL_STEP_NM: float = 3.0

_METADATA_COLS: tuple[str, ...] = (
    "Set",
    "Season",
    "Region",
    "Date",
    "Type",
    "Cultivar",
    "Pop",
    "Temp",
    "DM",
)
_TARGET_COL: str = "DM"
_GROUP_COL: str = "Season"

# safras esperadas (inteiros 1..5)
_EXPECTED_SEASONS: frozenset[int] = frozenset({1, 2, 3, 4, 5})


def load_mango_dmc_v3(
    csv_path: str | Path,
    keep_seasons: Iterable[int] | None = None,
) -> SpectralDataset:
    """Carrega Mango DMC v3 a partir do CSV do Mendeley.

    Parâmetros
    ----------
    csv_path : str | Path
        Caminho para `MangoDMC_NIR_Data_v3.csv` (~34 MB).
    keep_seasons : iterable de int, opcional
        Se fornecido, retorna apenas as amostras das safras especificadas
        (ex.: {1, 2, 3} para o pool de treino; {4} para o teste invariante;
        {5} para teste de generalização adicional). Padrão: todas as safras.

    Retorna
    -------
    SpectralDataset
        Tripla canônica (X, y, wavelengths) com metadados completos,
        sample_ids derivados e group_ids = Season.

    Erros
    -----
    FileNotFoundError
        Se `csv_path` não existir.
    ValueError
        Se o CSV não tiver a estrutura esperada, ou se `keep_seasons`
        contiver valores fora de {1, 2, 3, 4, 5}.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo Mango DMC v3 não encontrado: {csv_path}. "
            f"Baixe de https://data.mendeley.com/datasets/46htwnp833/3"
        )

    # validação do filtro de safras antes da leitura
    if keep_seasons is not None:
        keep_set = frozenset(int(s) for s in keep_seasons)
        invalidas = keep_set - _EXPECTED_SEASONS
        if invalidas:
            raise ValueError(
                f"keep_seasons contém safras inválidas: {sorted(invalidas)}. "
                f"Safras válidas: {sorted(_EXPECTED_SEASONS)}"
            )
    else:
        keep_set = None

    df = pd.read_csv(csv_path)

    # validação estrutural
    if df.shape[1] != _N_TOTAL_COLS:
        raise ValueError(
            f"Mango DMC: esperado {_N_TOTAL_COLS} colunas "
            f"({_N_METADATA_COLS} metadados + {_N_BANDS} bandas), "
            f"encontrado {df.shape[1]}"
        )
    cols_obs = tuple(df.columns[:_N_METADATA_COLS])
    if cols_obs != _METADATA_COLS:
        raise ValueError(
            f"Mango DMC: colunas de metadados esperadas {_METADATA_COLS}, "
            f"encontrado {cols_obs}"
        )

    # validação de comprimentos de onda (todos numéricos, passo 3 nm)
    band_col_names = list(df.columns[_N_METADATA_COLS:])
    wavelengths = _parse_wavelengths(band_col_names)
    _validate_wavelength_grid(wavelengths)

    # filtro opcional por safra
    if keep_set is not None:
        # garantia: Season deve ser inteiro
        seasons_in_file = df[_GROUP_COL].astype(int)
        mask = seasons_in_file.isin(keep_set)
        df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"Mango DMC: nenhuma amostra para keep_seasons={sorted(keep_set)}"
            )

    # extração de X, y, group_ids, sample_ids
    X = df.iloc[:, _N_METADATA_COLS:].to_numpy(dtype=np.float64)
    y = df[_TARGET_COL].to_numpy(dtype=np.float64)
    group_ids = df[_GROUP_COL].to_numpy(dtype=np.int64)
    sample_ids = _build_sample_ids(df)

    # detecta bandas sistematicamente zeradas (todos os valores == 0 em X)
    zero_mask = (X == 0).all(axis=0)
    zero_wavelengths = wavelengths[zero_mask].astype(int).tolist()

    metadata = {
        "domain": "mango_dmc_v3",
        "target_name": "DM",
        "target_unit": "%",  # dry matter content
        "wavelength_unit": "nm",
        "source": "https://data.mendeley.com/datasets/46htwnp833/3",
        "doi": "10.17632/46htwnp833.3",
        "wavelength_range_nm": (_WL_START_NM, _WL_END_NM),
        "wavelength_step_nm": _WL_STEP_NM,
        "instrument_note": (
            "Espectros NIR do mesocarpo de manga; região 285-1200 nm com "
            "passo de 3 nm. Bandas em comprimentos de onda fora da faixa "
            "útil do instrumento estão sistematicamente zeradas."
        ),
        "bands_systematically_zero": zero_wavelengths,
        "seasons_present": sorted(int(s) for s in np.unique(group_ids)),
        "n_samples_total_file": _N_SAMPLES_TOTAL,
        "n_samples_loaded": int(df.shape[0]),
        "keep_seasons_filter": (
            sorted(keep_set) if keep_set is not None else None
        ),
    }

    return SpectralDataset(
        X=X,
        y=y,
        wavelengths=wavelengths,
        metadata=metadata,
        sample_ids=sample_ids,
        group_ids=group_ids,
    )


# helpers --------------------------------------------------------------------


def _parse_wavelengths(col_names: list[str]) -> np.ndarray:
    """Converte ['285', '288', '291', ...] em np.ndarray float64."""
    try:
        return np.asarray([float(c) for c in col_names], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            f"Mango DMC: nome de coluna espectral não numérico. "
            f"Primeiros: {col_names[:5]}"
        ) from exc


def _validate_wavelength_grid(wavelengths: np.ndarray) -> None:
    """Valida faixa 285–1200 nm com passo de 3 nm."""
    if wavelengths[0] != _WL_START_NM:
        raise ValueError(
            f"Mango DMC: primeira banda esperada {_WL_START_NM} nm, "
            f"encontrada {wavelengths[0]} nm"
        )
    if wavelengths[-1] != _WL_END_NM:
        raise ValueError(
            f"Mango DMC: última banda esperada {_WL_END_NM} nm, "
            f"encontrada {wavelengths[-1]} nm"
        )
    diffs = np.diff(wavelengths)
    if not np.allclose(diffs, _WL_STEP_NM):
        raise ValueError(
            f"Mango DMC: passo de banda inconsistente; esperado {_WL_STEP_NM} nm. "
            f"Min={diffs.min()}, max={diffs.max()}"
        )


def _build_sample_ids(df: pd.DataFrame) -> np.ndarray:
    """Constrói sample_ids legíveis: 'S{Season}-{Cultivar}-Pop{Pop}-{linha}'.

    A {linha} usada é o índice DENTRO DO DATAFRAME corrente (já filtrado por
    safra, se aplicável), não a linha do arquivo original. Isso preserva a
    unicidade dos IDs após filtro.
    """
    seasons = df["Season"].astype(int).astype(str).to_numpy()
    cultivars = df["Cultivar"].astype(str).to_numpy()
    pops = df["Pop"].astype(int).astype(str).to_numpy()
    rows = np.arange(len(df)).astype(str)
    ids = np.array(
        [
            f"S{s}-{c}-Pop{p}-{r}"
            for s, c, p, r in zip(seasons, cultivars, pops, rows)
        ],
        dtype=object,
    )
    return ids
