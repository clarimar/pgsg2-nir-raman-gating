"""
Carregador do dataset Raman escolhido em ADR-0001 (pgsg_2):
bioprocess_substrates (Lange et al., 2026, Measurement), via
RamanBench/raman-data. Alvo único: concentração de glicose.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from pgsg2.ingestion.base import IngestionOperator, SpectralDataset

RAMANBENCH_DATASET_ID = "bioprocess_substrates"
TARGET_NAME = "glucose"


class RamanTargetNotFoundError(ValueError):
    """Lançado quando o alvo esperado não é encontrado em target_names."""


def _default_loader(dataset_id: str, cache_dir: Optional[str] = None) -> Any:
    import raman_data  # import tardio: pacote pesado, só necessário em uso real

    return raman_data.datasets.load_dataset(dataset_id, cache_dir=cache_dir)


def _find_target_index(target_names: list[str], target: str) -> int:
    target_lower = target.lower()
    matches = [i for i, name in enumerate(target_names) if target_lower in name.lower()]
    if not matches:
        raise RamanTargetNotFoundError(
            f"Alvo '{target}' não encontrado em target_names={target_names}. "
            "Verifique se o dataset RamanBench mudou de convenção de nomes."
        )
    if len(matches) > 1:
        raise RamanTargetNotFoundError(
            f"Mais de um alvo casa com '{target}': "
            f"{[target_names[i] for i in matches]}. Escolha precisa ser inequívoca."
        )
    return matches[0]


class RamanBioprocessSubstratesLoader(IngestionOperator):
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        loader_fn: Callable[[str, Optional[str]], Any] = _default_loader,
    ) -> None:
        self.cache_dir = cache_dir
        self._loader_fn = loader_fn

    def load(self) -> SpectralDataset:
        raw = self._loader_fn(RAMANBENCH_DATASET_ID, self.cache_dir)
        if raw is None:
            raise RuntimeError(
                f"Falha ao carregar '{RAMANBENCH_DATASET_ID}' via raman-data "
                "(load_dataset retornou None). Verifique conectividade/cache."
            )

        spectra = np.asarray(raw.spectra)
        raman_shifts = np.asarray(raw.raman_shifts)
        targets = np.asarray(raw.targets)
        target_names = list(raw.target_names)

        idx = _find_target_index(target_names, TARGET_NAME)
        y = targets[:, idx] if targets.ndim == 2 else targets

        valid = ~np.isnan(y)
        if not valid.all():
            n_dropped = int((~valid).sum())
            spectra = spectra[valid]
            y = y[valid]
        else:
            n_dropped = 0

        return SpectralDataset(
            X=spectra,
            y=y,
            wavelengths=raman_shifts,
            domain="raman",
            wavelength_unit="cm-1",
            target_name=TARGET_NAME,
            metadata={
                "source_dataset_id": RAMANBENCH_DATASET_ID,
                "source_package": "raman-data (RamanBench)",
                "n_samples_dropped_missing_target": n_dropped,
                "adr": "ADR-0001 (pgsg_2)",
            },
        )
