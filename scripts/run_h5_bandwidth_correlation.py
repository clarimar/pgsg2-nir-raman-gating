"""
Protocolo experimental H5 (pgsg_2): correlação estrutura-física.

Reformulação necessária (registrada em memória/discussão): a hipótese
original ("entropia e suavidade do gate correlacionam-se com a largura
de banda vibracional conhecida ... em cada modalidade") não é testável
como uma correlação estatística cross-modalidade, pois há apenas duas
modalidades (n=2). Este script opera a versão viável: DENTRO do Raman,
banda a banda, correlacionando a largura de linha real reportada na
literatura (patente US12372470, "Use of raman spectroscopy to monitor
culture medium") com três descritores LOCAIS extraídos do gate
realmente treinado (h1_raman_v2_result.npz) numa janela ao redor de
cada banda conhecida: FWHM local, entropia local, suavidade (TV) local.

NIR não entra nesse teste quantitativo: só há 3 bandas de literatura, e
o prior NIR já usa exatamente essas larguras (make_literature_prior),
tornando qualquer correlação circular por construção. Isso é reportado
como limitação, não escondido.

Não circular para Raman: RamanGlucosePrior usa sigma=15 cm-1 FIXO para
todas as bandas (Seção de priors), não as larguras reais variáveis
abaixo -- portanto, se o gate TREINADO (que parte desse prior uniforme)
mostrar estrutura local correlacionada com a largura real e variável
de cada banda, isso é um sinal genuíno aprendido dos dados, não herdado
do prior.

Uso:
    python scripts/run_h5_bandwidth_correlation.py
    (usa h1_raman_v2_result.npz, já gerado por run_h1_raman_v2.py)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

# (centro_cm1, largura_real_cm1) -- extraído da patente US12372470
# ("Use of raman spectroscopy to monitor culture medium"), largura =
# range_max - range_min reportado para cada banda de glicose.
LITERATURE_BAND_WIDTHS_CM1: tuple[tuple[float, float], ...] = (
    (402.0, 76.0),
    (527.0, 32.0),
    (589.0, 23.0),
    (732.0, 20.0),
    (789.0, 30.0),
    (855.0, 40.0),
    (911.0, 60.0),
    (968.0, 65.0),
    (1060.0, 20.0),
    (1073.0, 17.0),
    (1125.0, 30.0),
    (1155.0, 50.0),
    (1210.0, 50.0),
    (1276.0, 28.0),
    (1336.0, 12.0),
    (1371.0, 30.0),
    (1401.0, 20.0),
    (1450.0, 50.0),
    (1473.0, 15.0),
    (1549.0, 58.0),
)

WINDOW_HALF_WIDTH_CM1 = 40.0  # teto da janela; reduzido adaptativamente perto de vizinhos


def _adaptive_half_widths(centers: np.ndarray, max_half_width: float, margin: float = 0.45) -> np.ndarray:
    """
    Meia-largura de janela por banda, limitada para não invadir a banda
    vizinha mais próxima -- evita que a extração de FWHM de uma banda
    seja contaminada pelo pico de uma banda adjacente (ex.: 1060 e 1073
    cm-1 estão a só 13 cm-1 de distância).
    """
    centers = np.sort(centers)
    half_widths = np.full_like(centers, max_half_width, dtype=float)
    for i in range(len(centers)):
        neighbor_dists = [abs(centers[i] - centers[j]) for j in range(len(centers)) if j != i]
        nearest = min(neighbor_dists) if neighbor_dists else np.inf
        half_widths[i] = min(max_half_width, margin * nearest)
    return half_widths


def _local_window(wavelengths: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Índices dentro de [center-half_width, center+half_width]."""
    return np.where(np.abs(wavelengths - center) <= half_width)[0]


def _local_fwhm(g_window: np.ndarray, wl_window: np.ndarray) -> float:
    """FWHM empírica do gate dentro da janela: largura (em cm-1) onde
    g excede metade do pico local (baseline = mínimo da janela)."""
    if len(g_window) < 3:
        return float("nan")
    baseline = g_window.min()
    peak = g_window.max()
    half = baseline + (peak - baseline) / 2
    above = g_window >= half
    if not above.any():
        return float("nan")
    idx_above = np.where(above)[0]
    return float(wl_window[idx_above[-1]] - wl_window[idx_above[0]])


def _local_entropy(g_window: np.ndarray) -> float:
    """Entropia de Shannon bruta (não normalizada) -- reportada só para
    referência; a métrica usada na correlação é _local_entropy_normalized."""
    g = np.clip(g_window, 1e-12, None)
    g_norm = g / g.sum()
    return float(-np.sum(g_norm * np.log(g_norm)))


def _local_entropy_normalized(g_window: np.ndarray) -> float:
    """
    Entropia de Shannon NORMALIZADA pelo log(n) da janela: H(g)/log(n).

    Correção do confundidor identificado após a primeira rodada real:
    a janela adaptativa tem tamanho (número de pontos) diferente por
    banda, e a entropia bruta de Shannon cresce com o número de pontos
    amostrados mesmo sem nenhuma mudança na forma do gate (o teto
    log(n) sobe). Dividir por log(n) mede "quão perto do máximo possível
    para aquele n" a distribuição está -- comparável entre janelas de
    tamanhos diferentes.
    """
    n = len(g_window)
    if n < 2:
        return float("nan")
    raw = _local_entropy(g_window)
    max_entropy = np.log(n)
    if max_entropy < 1e-12:
        return float("nan")
    return float(raw / max_entropy)


def _local_smoothness(g_window: np.ndarray) -> float:
    if len(g_window) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(g_window))))


def main():
    try:
        data = np.load("h1_raman_v2_result.npz")
    except FileNotFoundError:
        print("[ERRO] h1_raman_v2_result.npz não encontrado -- rode scripts/run_h1_raman_v2.py primeiro.")
        return

    gate = data["gate"]
    wavelengths = data["wavelengths"]

    centers, real_widths, fwhms, entropies, entropies_norm, smoothnesses = [], [], [], [], [], []

    print("=" * 78)
    print("H5 (Raman): estrutura local do gate vs. largura de banda real (literatura)")
    print("=" * 78)
    print(f"{'centro (cm-1)':>14} {'largura real':>13} {'janela (±cm-1)':>15} {'n pts':>6} {'FWHM gate':>11} {'entropia (H/logN)':>18} {'suavidade local':>16}")

    centers_arr = np.array([c for c, _ in LITERATURE_BAND_WIDTHS_CM1])
    half_widths_adaptive = _adaptive_half_widths(centers_arr, WINDOW_HALF_WIDTH_CM1)

    for (center, real_width), half_width in zip(LITERATURE_BAND_WIDTHS_CM1, half_widths_adaptive):
        idx = _local_window(wavelengths, center, half_width)
        if len(idx) < 3:
            continue
        g_win = gate[idx]
        wl_win = wavelengths[idx]

        fwhm = _local_fwhm(g_win, wl_win)
        entropy = _local_entropy(g_win)
        entropy_norm = _local_entropy_normalized(g_win)
        smoothness = _local_smoothness(g_win)

        centers.append(center)
        real_widths.append(real_width)
        fwhms.append(fwhm)
        entropies.append(entropy)
        entropies_norm.append(entropy_norm)
        smoothnesses.append(smoothness)

        print(f"{center:>14.1f} {real_width:>13.1f} {half_width:>15.1f} {len(idx):>6d} {fwhm:>11.2f} {entropy_norm:>18.4f} {smoothness:>16.6f}")

    real_widths = np.array(real_widths)
    fwhms = np.array(fwhms)
    entropies = np.array(entropies)
    entropies_norm = np.array(entropies_norm)
    smoothnesses = np.array(smoothnesses)

    valid = ~np.isnan(fwhms)
    n_valid = valid.sum()

    print()
    print(f"n bandas válidas: {n_valid} (de {len(LITERATURE_BAND_WIDTHS_CM1)})")

    def _report_corr(name, x, y):
        if len(x) < 4:
            print(f"  {name}: n insuficiente para correlação")
            return
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)
        print(f"  {name}: Pearson r={r_pearson:+.3f} (p={p_pearson:.3f})   "
              f"Spearman rho={r_spearman:+.3f} (p={p_spearman:.3f})")

    print("\nCorrelação: largura real (literatura) vs. descritores locais do gate")
    _report_corr("largura real vs. FWHM do gate                ", real_widths[valid], fwhms[valid])
    _report_corr("largura real vs. entropia normalizada (H/logN)", real_widths[valid], entropies_norm[valid])
    _report_corr("largura real vs. entropia bruta (referência)  ", real_widths[valid], entropies[valid])
    _report_corr("largura real vs. suavidade local               ", real_widths[valid], smoothnesses[valid])

    np.savez(
        "h5_bandwidth_correlation_result.npz",
        centers=np.array(centers), real_widths=real_widths,
        fwhms=fwhms, entropies=entropies, entropies_normalized=entropies_norm,
        smoothnesses=smoothnesses,
    )
    print("\nResultado salvo em h5_bandwidth_correlation_result.npz")
    print("\nNOTA: NIR não entra neste teste quantitativo -- apenas 3 bandas de")
    print("literatura, e o prior NIR já usa exatamente essas larguras, tornando")
    print("qualquer correlação circular por construção. Ver docstring do script.")


if __name__ == "__main__":
    main()
