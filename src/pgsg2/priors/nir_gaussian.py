"""
Prior NIR de literatura em forma GAUSSIANA (ablação de forma funcional).

O prior oficial de pgsg_1 (``make_literature_prior``, em ``pgsg_v2.py``)
usa uma função DEGRAU: patamares constantes por região, com saltos
abruptos nas fronteiras (680/700/900/1000/1100/1200 nm). Isso não é
modificado -- é o prior que gerou os resultados publicados e continua
sendo o oficial para H1.

Este módulo existe só para uma ablação de H3 (suavidade do gate):
recodifica as MESMAS bandas literárias (mesmos centros, mesmos pesos
relativos) como gaussianas suaves, no mesmo estilo do
``RamanGlucosePrior``, para permitir uma comparação de topologia
NIR-vs-Raman sem o confundidor "prior degrau vs. prior gaussiano" --
ver achado registrado após a extração dos gates reais (rho muito alto
em ambas as modalidades, então a suavidade do gate final herda a
suavidade da forma funcional do prior).

Bandas e pesos idênticos a make_literature_prior:
    - 680-700 nm (centro ~690 nm): clorofila, peso 0.6
    - 900-1000 nm (centro ~950 nm): 2o sobretom C-H, peso 1.0
    - 1100-1200 nm (centro ~1150 nm): 1o sobretom C-H, peso 0.8
    - baseline 0.1 nas demais bandas
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (centro_nm, peso, sigma_nm) -- sigma escolhido para que a maior parte
# da massa da gaussiana caia dentro da faixa original de 20 nm de largura
# (meia-largura ~10 nm -> sigma ~10 nm mantém ~68% da massa dentro da
# faixa original, análogo em espírito ao patamar original).
NIR_LITERATURE_BANDS_NM: tuple[tuple[float, float, float], ...] = (
    (690.0, 0.6, 10.0),   # clorofila
    (950.0, 1.0, 50.0),   # 2o sobretom C-H (faixa original mais larga: 100nm)
    (1150.0, 0.8, 50.0),  # 1o sobretom C-H (faixa original mais larga: 100nm)
)

NIR_BASELINE: float = 0.1


@dataclass
class NIRLiteratureGaussianPrior:
    """Prior NIR de literatura, versão gaussiana (ablação de H3).

    Mesmos centros e pesos relativos de ``make_literature_prior``,
    codificados como soma de gaussianas em vez de patamares constantes.
    """

    bands: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: NIR_LITERATURE_BANDS_NM
    )
    baseline: float = NIR_BASELINE
    floor: float = 0.01

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        wavelengths = np.asarray(wavelengths, dtype=float)
        prior = np.full_like(wavelengths, self.baseline)

        for center, weight, sigma in self.bands:
            bump = weight * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
            prior = np.maximum(prior, self.baseline + bump)

        peak = prior.max()
        if peak > 0:
            prior = prior / peak
        return np.clip(prior, self.floor, 1.0)
