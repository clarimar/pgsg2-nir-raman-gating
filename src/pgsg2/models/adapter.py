"""
Adaptador de contrato: pgsg2.ingestion.base.SpectralDataset ->
pgsg_1.ingestion.SpectralDataset.

Este módulo NUNCA modifica pgsg_1. Ele existe justamente para que o
operador M de pgsg_2 possa chamar PGSGModel/PLSModel de pgsg_1 sem
qualquer alteração de código -- condição de teste da Hipótese 1
("sem alteração arquitetural").

Os dois contratos SpectralDataset divergem:
    - pgsg2: domain/wavelength_unit são atributos de primeira classe;
      metadata é livre.
    - pgsg_1: domain/target_name/target_unit/wavelength_unit/source
      vivem dentro de metadata (chaves obrigatórias, REQUIRED_METADATA_KEYS);
      o dataclass é frozen.

A conversão é sempre pgsg2 -> pgsg_1, nunca o inverso: pgsg_1 não
precisa saber que pgsg2 existe.
"""

from __future__ import annotations

from typing import Optional

from pgsg_1.ingestion import SpectralDataset as PGSG1SpectralDataset

from pgsg2.ingestion.base import SpectralDataset as PGSG2SpectralDataset


def to_pgsg1_dataset(
    ds: PGSG2SpectralDataset,
    *,
    target_unit: str = "unknown",
    source: Optional[str] = None,
) -> PGSG1SpectralDataset:
    """Converte um SpectralDataset de pgsg2 para o contrato de pgsg_1.

    Args:
        ds: dataset no contrato de pgsg2 (com domain/wavelength_unit
            como atributos de primeira classe).
        target_unit: unidade do alvo (ex.: "g/L"). Default "unknown"
            quando a unidade original não está confirmada na
            documentação da fonte -- ver ADR-0001 (glicose em
            bioprocess_substrates: unidade a confirmar).
        source: proveniência explícita; se None, é derivada de
            ds.metadata (source_dataset_id + source_package).

    Returns:
        Instância de pgsg_1.ingestion.SpectralDataset, com todos os
        invariantes I1-I4 daquele contrato validados no construtor.
    """
    if source is None:
        parts = [
            str(ds.metadata.get("source_dataset_id", "unknown")),
            str(ds.metadata.get("source_package", "")),
        ]
        source = " via ".join(p for p in parts if p)

    metadata = {
        "domain": ds.domain,
        "target_name": ds.target_name,
        "target_unit": target_unit,
        "wavelength_unit": ds.wavelength_unit,
        "source": source,
        # metadados extras de pgsg2 preservados para rastreabilidade,
        # além das 5 chaves obrigatórias de pgsg_1.
        **{
            f"pgsg2_{k}": v
            for k, v in ds.metadata.items()
            if k not in ("source_dataset_id", "source_package")
        },
    }

    return PGSG1SpectralDataset(
        X=ds.X,
        y=ds.y,
        wavelengths=ds.wavelengths,
        metadata=metadata,
    )
