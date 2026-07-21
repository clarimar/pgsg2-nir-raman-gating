"""
Operador P (prior) do ramo Raman de pgsg_2: prior fisicamente motivado
por atribuições vibracionais da literatura para glicose.

Princípio anti-circularidade (mesmo estabelecido em pgsg_1): o prior é
construído *apenas* a partir do eixo espectral (raman_shifts) e de
conhecimento publicado sobre onde a glicose absorve/espalha -- nunca a
partir de X ou y do experimento em execução. A assinatura de
``RamanGlucosePrior.compute`` reflete isso: recebe só ``wavelengths``.

Fontes literárias (ver docstring de GLUCOSE_RAMAN_BANDS_CM1):
    - Lyu et al., "In Vivo Blood Glucose Quantification Using Raman
      Spectroscopy", PLOS ONE, 2012: bandas 911, 1060, 1125 cm-1
      reportadas como "as impressões digitais Raman da glicose".
    - Kang et al., "Direct observation of glucose fingerprint using in
      vivo Raman spectroscopy", Science Advances, 2020: mesmas 3 bandas
      (911, 1060, 1125 cm-1) usadas em regressão multi-linear.
    - Deng et al., "Determination of low concentration glucose solution
      using Raman spectroscopy...", Results in Physics, 2025: DFT
      confirma 1125 cm-1 como banda característica da glicopiranose.
    - US Patent 11249026 ("Use of raman spectroscopy to monitor culture
      medium"): lista estendida de bandas de glicose *especificamente
      em meio de cultura de bioprocesso* -- contexto direto do dataset
      bioprocess_substrates usado neste projeto (ADR-0001).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# Bandas com múltiplas fontes independentes convergentes (alta confiança).
# Atribuição estrutural: região de estiramento C-O/C-C do anel piranose
# da glicose (fingerprint de carboidratos, 900-1150 cm-1).
GLUCOSE_RAMAN_BANDS_PRIMARY_CM1: tuple[float, ...] = (911.0, 1060.0, 1125.0)

# Bandas adicionais, de uma única fonte (contexto de meio de cultura de
# bioprocesso), usadas com peso menor -- confiança mais baixa que as
# primárias, mas diretamente relevantes ao domínio do dataset escolhido.
GLUCOSE_RAMAN_BANDS_SECONDARY_CM1: tuple[float, ...] = (
    402.0, 527.0, 589.0, 732.0, 789.0, 855.0, 968.0,
    1155.0, 1210.0, 1276.0, 1336.0, 1371.0, 1401.0, 1450.0, 1473.0, 1549.0,
)


class PriorOperator(ABC):
    """Contrato do operador P: constrói um prior a partir do eixo espectral."""

    @abstractmethod
    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        """Devolve um vetor de prior, mesmo shape de ``wavelengths``,
        com valores em (floor, 1]."""
        raise NotImplementedError


@dataclass
class RamanGlucosePrior(PriorOperator):
    """
    Prior gaussiano multi-banda para glicose em Raman.

    Cada banda literária vira uma gaussiana centrada em seu número de
    onda, com largura ``sigma_cm1`` (aproximação da largura de linha
    típica de bandas Raman de carboidratos em solução/meio de cultura,
    tipicamente 10-20 cm-1). O prior final é a soma ponderada dessas
    gaussianas, normalizada para o intervalo (floor, 1].

    Attributes:
        sigma_cm1: largura (desvio-padrão) das gaussianas, em cm-1.
        floor: valor mínimo do prior (evita zeros exatos, que
            inviabilizariam a regularização KL gate-prior caso o valor
            do prior apareça no denominador ou dentro de um log).
        primary_weight: peso das bandas de alta confiança (múltiplas
            fontes independentes).
        secondary_weight: peso das bandas de confiança mais baixa
            (fonte única, contexto de meio de cultura).
    """

    sigma_cm1: float = 15.0
    floor: float = 0.01
    primary_weight: float = 1.0
    secondary_weight: float = 0.4
    primary_bands: tuple[float, ...] = field(
        default_factory=lambda: GLUCOSE_RAMAN_BANDS_PRIMARY_CM1
    )
    secondary_bands: tuple[float, ...] = field(
        default_factory=lambda: GLUCOSE_RAMAN_BANDS_SECONDARY_CM1
    )

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        wavelengths = np.asarray(wavelengths, dtype=float)
        prior = np.zeros_like(wavelengths)

        for center in self.primary_bands:
            prior += self.primary_weight * np.exp(
                -0.5 * ((wavelengths - center) / self.sigma_cm1) ** 2
            )
        for center in self.secondary_bands:
            prior += self.secondary_weight * np.exp(
                -0.5 * ((wavelengths - center) / self.sigma_cm1) ** 2
            )

        peak = prior.max()
        if peak > 0:
            prior = prior / peak
        return np.clip(prior, self.floor, 1.0)
