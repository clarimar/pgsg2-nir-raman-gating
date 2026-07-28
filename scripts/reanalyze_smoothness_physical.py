"""
Reanálise pós-hoc de H3: suavidade do gate normalizada pela densidade
FÍSICA de bandas (cm-1), não pelo índice bruto -- testa se a diferença
NIR-vs-Raman é um artefato de "quantas bandas cabem por unidade física"
(NIR: p=281 sobre ~915 nm; Raman: p=1870 sobre ~2995 cm-1).

Não requer retreinamento -- usa os .npz já salvos por:
    scripts/run_h1_raman_v2.py       -> h1_raman_v2_result.npz
    scripts/ablation_h3_smoothness.py -> ablation_h3_result.npz
    scripts/extract_nir_gate_v2.py    -> nir_gate_v2_extraction.npz

Conversão nm -> cm-1: nu_tilde = 1e7 / lambda_nm (necessária porque
comprimento de onda e número de onda não se relacionam por um fator de
escala fixo -- é uma conversão não linear, 1/lambda).

Uso:
    python scripts/reanalyze_smoothness_physical.py
"""

from __future__ import annotations

import numpy as np

from pgsg2.interpretability.topology import gate_smoothness, gate_smoothness_physical


def nm_to_cm1(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Converte nm para número de onda (cm-1): nu_tilde = 1e7 / lambda_nm."""
    return 1e7 / np.asarray(wavelengths_nm, dtype=float)


def main():
    print("=" * 70)
    print("Reanálise: suavidade do gate normalizada por densidade física de bandas")
    print("=" * 70)

    # ---- Raman (prior gaussiano, resultado de H1 real) ----
    try:
        raman = np.load("h1_raman_v2_result.npz")
        g_raman = raman["gate"]
        wl_raman_cm1 = raman["wavelengths"]  # já em cm-1

        tv_raw_raman = gate_smoothness(g_raman)
        tv_phys_raman = gate_smoothness_physical(g_raman, wl_raman_cm1)

        print(f"\nRaman (bioprocess_substrates, prior gaussiano):")
        print(f"  p = {len(g_raman)}, faixa = [{wl_raman_cm1.min():.1f}, {wl_raman_cm1.max():.1f}] cm-1")
        print(f"  TV bruta (por índice)      = {tv_raw_raman:.6f}")
        print(f"  TV física (por cm-1)       = {tv_phys_raman:.8f}")
    except FileNotFoundError:
        print("\n[AVISO] h1_raman_v2_result.npz não encontrado -- rode scripts/run_h1_raman_v2.py primeiro.")
        return

    # ---- NIR (prior gaussiano, ablação) ----
    try:
        nir_gauss = np.load("ablation_h3_result.npz")
        g_nir = nir_gauss["gate_nir_gauss"]
        wl_nir_nm = nir_gauss["wavelengths_nir"]  # em nm
        wl_nir_cm1 = nm_to_cm1(wl_nir_nm)
        # nm->cm-1 inverte a ordem (comprimento de onda maior = numero de onda menor);
        # reordena para manter eixo crescente, mantendo g e wl pareados
        order = np.argsort(wl_nir_cm1)
        g_nir_sorted = g_nir[order]
        wl_nir_cm1_sorted = wl_nir_cm1[order]

        tv_raw_nir = gate_smoothness(g_nir)  # ordem original nm, TV bruta não depende de direção
        tv_phys_nir = gate_smoothness_physical(g_nir_sorted, wl_nir_cm1_sorted)

        print(f"\nNIR (Mango DMC v3 Safra 4, prior gaussiano):")
        print(f"  p = {len(g_nir)}, faixa = [{wl_nir_nm.min():.1f}, {wl_nir_nm.max():.1f}] nm "
              f"= [{wl_nir_cm1.min():.1f}, {wl_nir_cm1.max():.1f}] cm-1")
        print(f"  TV bruta (por índice)      = {tv_raw_nir:.6f}")
        print(f"  TV física (por cm-1, convertido de nm) = {tv_phys_nir:.8f}")
    except FileNotFoundError:
        print("\n[AVISO] ablation_h3_result.npz não encontrado -- rode scripts/ablation_h3_smoothness.py primeiro.")
        return

    print()
    print("=" * 70)
    print("COMPARAÇÃO (mesma unidade física: variação de gate por cm-1)")
    print("=" * 70)
    print(f"TV bruta (por índice):  NIR={tv_raw_nir:.6f}   Raman={tv_raw_raman:.6f}   "
          f"NIR{'>' if tv_raw_nir > tv_raw_raman else '<'}Raman")
    print(f"TV física (por cm-1):   NIR={tv_phys_nir:.8f}   Raman={tv_phys_raman:.8f}   "
          f"NIR{'>' if tv_phys_nir > tv_phys_raman else '<'}Raman")
    print()
    if (tv_raw_nir > tv_raw_raman) == (tv_phys_nir > tv_phys_raman):
        print("A DIREÇÃO SE MANTÉM após normalizar pela densidade física de bandas.")
        print("-> A diferença de suavidade do gate NÃO parece ser um artefato de")
        print("   quantas bandas cabem por unidade física (confundidor descartado).")
    else:
        print("A DIREÇÃO SE INVERTE após normalizar pela densidade física de bandas!")
        print("-> A diferença de suavidade do gate ERA, ao menos em parte, um artefato")
        print("   de densidade de amostragem espectral, não uma diferença física real.")


if __name__ == "__main__":
    main()
