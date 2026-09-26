"""ingestion — operador G do pipeline.

Contrato (do documento de pipeline, seção 3)
--------------------------------------------
Para cada base, G produz uma tripla canonizada (X, y, lambda) com

    X in R^{n x p}     matriz espectral
    y in R^{n}         alvo contínuo
    lambda in R^{p}    vetor de comprimentos de onda

Metadados essenciais (unidade de y, faixa espectral, identificador de
domínio, identificador de safra/instrumento quando aplicável) acompanham
a tripla — são necessários na síntese para distinguir efeitos de escala
de efeitos de domínio.

Bases suportadas
----------------
- tecator       (215 amostras, 100 bandas, 850–1050 nm, gordura) — IMPLEMENTADO
- gasoline      (60 amostras, 401 bandas, 900–1700 nm, octanagem) — IMPLEMENTADO
- mango_dmc_v3  (12.011 amostras, 306 bandas, 285–1200 nm, DM) — IMPLEMENTADO

API pública
-----------
- SpectralDataset:        tipo canônico de saída (dataclass imutável)
- load_tecator:           loader Tecator
- load_gasoline:          loader Gasoline
- load_mango_dmc_v3:      loader Mango DMC v3 (espinha dorsal da escala)

Invariantes (testados em tests/test_ingestion_*.py)
---------------------------------------------------
I1. X.shape == (n, p), y.shape == (n,), lambda.shape == (p,)
I2. lambda estritamente crescente
I3. nenhum NaN em X, y, lambda
I4. metadados obrigatórios presentes (ver REQUIRED_METADATA_KEYS)
"""

from .dataset import REQUIRED_METADATA_KEYS, SpectralDataset
from .gasoline import load_gasoline
from .mango import load_mango_dmc_v3
from .tecator import load_tecator

__all__ = [
    "SpectralDataset",
    "REQUIRED_METADATA_KEYS",
    "load_tecator",
    "load_gasoline",
    "load_mango_dmc_v3",
]
