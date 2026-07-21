"""
Testes de contrato do operador P (prior) para o ramo Raman.

Verificam: (i) range válido, (ii) máximos locais nas bandas literárias
conhecidas, (iii) que a assinatura de `compute` não aceita X/y --
garantia estrutural do princípio anti-circularidade --, (iv) invariância
a reordenação do eixo espectral.
"""

import inspect

import numpy as np
import pytest

from pgsg2.priors.raman import (
    GLUCOSE_RAMAN_BANDS_PRIMARY_CM1,
    GLUCOSE_RAMAN_BANDS_SECONDARY_CM1,
    RamanGlucosePrior,
)


def test_prior_signature_has_no_data_dependent_arguments():
    """
    Garantia estrutural anti-circularidade: `compute` só pode receber o
    eixo espectral, nunca X ou y do experimento -- isso é verificado na
    própria assinatura do método, não apenas por convenção.
    """
    sig = inspect.signature(RamanGlucosePrior.compute)
    params = list(sig.parameters.keys())
    assert params == ["self", "wavelengths"]


def test_prior_values_within_valid_range():
    wavelengths = np.linspace(390, 3385, 1870)
    prior = RamanGlucosePrior().compute(wavelengths)
    assert prior.shape == wavelengths.shape
    assert (prior > 0).all()
    assert (prior <= 1.0).all()


def test_prior_peaks_at_primary_bands():
    wavelengths = np.linspace(390, 3385, 3000)
    prior = RamanGlucosePrior().compute(wavelengths)

    for center in GLUCOSE_RAMAN_BANDS_PRIMARY_CM1:
        idx_center = np.argmin(np.abs(wavelengths - center))
        idx_far = np.argmin(np.abs(wavelengths - (center + 500)))  # região distante
        assert prior[idx_center] > prior[idx_far]


def test_primary_bands_stronger_than_secondary_only_bands():
    """As bandas primárias (múltiplas fontes) devem gerar prior mais alto
    do que uma banda secundária isolada, na mesma configuração."""
    wavelengths = np.linspace(390, 3385, 3000)
    prior = RamanGlucosePrior().compute(wavelengths)

    idx_primary = np.argmin(np.abs(wavelengths - GLUCOSE_RAMAN_BANDS_PRIMARY_CM1[-1]))  # 1125
    idx_secondary = np.argmin(np.abs(wavelengths - GLUCOSE_RAMAN_BANDS_SECONDARY_CM1[0]))  # 402

    assert prior[idx_primary] > prior[idx_secondary]


def test_prior_floor_prevents_exact_zero():
    wavelengths = np.linspace(390, 3385, 1870)
    prior = RamanGlucosePrior(floor=0.02).compute(wavelengths)
    assert (prior >= 0.02).all()


def test_prior_is_deterministic_and_order_independent_per_wavelength():
    wavelengths = np.linspace(390, 3385, 500)
    prior_a = RamanGlucosePrior().compute(wavelengths)

    shuffled_idx = np.random.default_rng(0).permutation(len(wavelengths))
    prior_b = RamanGlucosePrior().compute(wavelengths[shuffled_idx])

    np.testing.assert_allclose(prior_a[shuffled_idx], prior_b)


def test_prior_handles_wavelength_range_outside_all_bands():
    # Eixo hipotético que não cobre nenhuma banda conhecida de glicose.
    wavelengths = np.linspace(2000, 2100, 200)
    prior = RamanGlucosePrior().compute(wavelengths)
    assert prior.shape == wavelengths.shape
    assert (prior > 0).all()  # floor garante positividade mesmo sem pico
