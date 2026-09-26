"""Preprocessor — implementação do operador T.

Segue o padrão fit/transform do scikit-learn (ADR de design confirmado
em conversa): `.fit(treino)` estima os parâmetros phi EXCLUSIVAMENTE no
treino; `.transform(dataset)` reaplica os mesmos parâmetros a qualquer
conjunto (treino ou teste) — Princípio da Higiene de Vazamento.

Ordem fixa das sub-transformações (decisão de design):
    1. descarte de bandas zeradas (automático via metadata, opção de desligar)
    2. Savitzky-Golay (opcional, default desligado, parâmetros obrigatórios)
    3. SNV (por linha; corrige espalhamento)
    4. Z-score por banda (mu, sigma do treino)

O alvo y também é Z-score-normalizado; os parâmetros (mu_y, sigma_y)
ficam guardados em phi para que o estágio de avaliação possa desnormalizar
antes de calcular métricas na escala original.

Sobre safety_datasets (ADR-0005)
--------------------------------
O `fit` aceita opcionalmente uma lista de datasets adicionais
("safety_datasets") para os quais a faixa útil deve ser válida.
A máscara de bandas a manter resulta da INTERSEÇÃO entre as faixas úteis
do treino e dos safety_datasets — equivalentemente, a UNIÃO das bandas
zeradas entre todos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import savgol_filter

from pgsg_1.ingestion import SpectralDataset

# tolerância numérica para divisões por desvio-padrão ~0
_EPS: float = 1e-8

# tolerância para casar comprimentos de onda zerados com as colunas de lambda
_WL_MATCH_TOLERANCE_NM: float = 0.5


@dataclass(frozen=True)
class PreprocessorParams:
    """Os parâmetros ajustados (o 'phi' do contrato).

    Imutável: uma vez ajustado no treino, não muda. Guarda tudo o que é
    necessário para reaplicar a transformação de forma idêntica.

    Atributos
    ---------
    keep_band_mask : np.ndarray de bool, shape (p_original,)
        Máscara das bandas mantidas (True = mantém). Bandas zeradas no
        treino OU em qualquer safety_dataset ficam False.
    kept_wavelengths : np.ndarray, shape (p_kept,)
        Comprimentos de onda das bandas mantidas (subconjunto de lambda).
    apply_sg : bool
        Se Savitzky-Golay foi aplicado.
    sg_window : int | None
        Janela do SG (ímpar). None se apply_sg=False.
    sg_polyorder : int | None
        Ordem do polinômio do SG. None se apply_sg=False.
    sg_deriv : int | None
        Ordem da derivada do SG (0 = só suavização). None se apply_sg=False.
    apply_snv : bool
        Se SNV foi aplicado.
    mu_bands : np.ndarray, shape (p_kept,)
        Médias por banda (estimadas no treino, PÓS-SG e PÓS-SNV se ativos).
    sigma_bands : np.ndarray, shape (p_kept,)
        Desvios por banda (idem).
    mu_y : float
        Média do alvo no treino.
    sigma_y : float
        Desvio do alvo no treino.
    n_safety_datasets : int
        Quantos safety_datasets foram usados no fit (informativo).
    """

    keep_band_mask: np.ndarray
    kept_wavelengths: np.ndarray
    apply_sg: bool
    sg_window: int | None
    sg_polyorder: int | None
    sg_deriv: int | None
    apply_snv: bool
    mu_bands: np.ndarray
    sigma_bands: np.ndarray
    mu_y: float
    sigma_y: float
    n_safety_datasets: int = 0


class Preprocessor:
    """Operador T: pré-processamento espectral com higiene de vazamento.

    Parâmetros
    ----------
    drop_zero_bands : bool
        Se True (padrão), descarta automaticamente as bandas listadas em
        `metadata['bands_systematically_zero']` do(s) dataset(s).
    apply_sg : bool
        Se True, aplica Savitzky-Golay (suavização ± derivada). Default False.
        Quando True, os parâmetros sg_window, sg_polyorder e sg_deriv são
        OBRIGATÓRIOS — não há defaults, cada experimento deve escolher
        conscientemente.
    sg_window : int | None
        Largura da janela do SG. Deve ser inteiro positivo ÍMPAR.
        Só relevante se apply_sg=True.
    sg_polyorder : int | None
        Grau do polinômio ajustado em cada janela. Deve ser <  sg_window.
        Só relevante se apply_sg=True.
    sg_deriv : int | None
        Ordem da derivada (0 = só suavização). Deve ser <= sg_polyorder.
        Só relevante se apply_sg=True.
    apply_snv : bool
        Se True (padrão), aplica SNV por espectro.
    normalize_target : bool
        Se True (padrão), Z-score-normaliza o alvo y.

    Uso simples (treino só, sem SG)
    -------------------------------
    >>> pre = Preprocessor()
    >>> pre.fit(ds_treino)
    >>> X_tr, y_tr = pre.transform(ds_treino)

    Uso com Savitzky-Golay (replicando Anderson et al. 2020 na Mango)
    -----------------------------------------------------------------
    >>> pre = Preprocessor(apply_sg=True, sg_window=17, sg_polyorder=2, sg_deriv=2)
    >>> pre.fit(ds_treino, safety_datasets=[ds_teste])
    >>> X_tr, y_tr = pre.transform(ds_treino)
    >>> X_te, y_te = pre.transform(ds_teste)
    """

    def __init__(
        self,
        drop_zero_bands: bool = True,
        apply_sg: bool = False,
        sg_window: int | None = None,
        sg_polyorder: int | None = None,
        sg_deriv: int | None = None,
        apply_snv: bool = True,
        normalize_target: bool = True,
    ) -> None:
        # validação dos parâmetros de SG ANTES do fit (falha rápido)
        if apply_sg:
            self._validate_sg_params(sg_window, sg_polyorder, sg_deriv)

        self.drop_zero_bands = drop_zero_bands
        self.apply_sg = apply_sg
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.sg_deriv = sg_deriv
        self.apply_snv = apply_snv
        self.normalize_target = normalize_target
        self._params: PreprocessorParams | None = None

    @staticmethod
    def _validate_sg_params(
        window: int | None, polyorder: int | None, deriv: int | None
    ) -> None:
        """Valida os três parâmetros do Savitzky-Golay.

        Regras (scipy.signal.savgol_filter):
        - window: inteiro positivo ímpar
        - polyorder: 0 <= polyorder < window
        - deriv: 0 <= deriv <= polyorder
        """
        if window is None or polyorder is None or deriv is None:
            raise ValueError(
                "apply_sg=True exige sg_window, sg_polyorder e sg_deriv "
                "(sem defaults; cada experimento deve escolher)."
            )
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"sg_window deve ser inteiro positivo; recebido {window!r}")
        if window % 2 == 0:
            raise ValueError(f"sg_window deve ser ÍMPAR; recebido {window}")
        if not isinstance(polyorder, int) or polyorder < 0:
            raise ValueError(
                f"sg_polyorder deve ser inteiro >= 0; recebido {polyorder!r}"
            )
        if polyorder >= window:
            raise ValueError(
                f"sg_polyorder ({polyorder}) deve ser < sg_window ({window})"
            )
        if not isinstance(deriv, int) or deriv < 0:
            raise ValueError(f"sg_deriv deve ser inteiro >= 0; recebido {deriv!r}")
        if deriv > polyorder:
            raise ValueError(
                f"sg_deriv ({deriv}) deve ser <= sg_polyorder ({polyorder})"
            )

    # fit --------------------------------------------------------------------

    def fit(
        self,
        dataset: SpectralDataset,
        safety_datasets: Sequence[SpectralDataset] | None = None,
    ) -> "Preprocessor":
        """Estima phi a partir do dataset de TREINO.

        Parâmetros
        ----------
        dataset : SpectralDataset
            O dataset de TREINO. Estatísticas mu/sigma são estimadas SÓ daqui.
        safety_datasets : sequência de SpectralDataset, opcional
            Datasets adicionais (tipicamente o teste invariante) cujas
            bandas zeradas também devem ser descartadas. Ver ADR-0005.
        """
        if safety_datasets is None:
            safety_datasets = []
        else:
            safety_datasets = list(safety_datasets)

        # validação: safety_datasets devem ter mesma grade do treino
        for i, sd in enumerate(safety_datasets):
            if sd.X.shape[1] != dataset.X.shape[1]:
                raise ValueError(
                    f"safety_datasets[{i}] tem {sd.X.shape[1]} bandas, "
                    f"treino tem {dataset.X.shape[1]}"
                )
            if not np.allclose(sd.wavelengths, dataset.wavelengths):
                raise ValueError(
                    f"safety_datasets[{i}] tem grade de comprimentos de onda "
                    f"diferente do treino. Os datasets devem ser da mesma fonte."
                )

        # 1. máscara de bandas a manter (interseção via união dos zeros)
        keep_mask = self._compute_keep_mask(dataset, safety_datasets)
        kept_wl = dataset.wavelengths[keep_mask]

        # aplica descarte de bandas
        X = dataset.X[:, keep_mask]

        # validação adicional: SG exige que a janela caiba na largura espectral
        if self.apply_sg and X.shape[1] < self.sg_window:
            raise ValueError(
                f"Após descarte de bandas, restam apenas {X.shape[1]} bandas — "
                f"insuficientes para sg_window={self.sg_window}. "
                f"Reduza a janela ou afrouxe o descarte."
            )

        # 2. Savitzky-Golay (suavização espectral, por amostra)
        if self.apply_sg:
            X = self._apply_sg(X)

        # 3. SNV (por linha) — após SG
        if self.apply_snv:
            X = self._snv(X)

        # 4. Z-score por banda: estima mu, sigma NO TREINO (não nos safety)
        mu_bands = X.mean(axis=0)
        sigma_bands = X.std(axis=0)

        # alvo
        if self.normalize_target:
            mu_y = float(dataset.y.mean())
            sigma_y = float(dataset.y.std())
        else:
            mu_y = 0.0
            sigma_y = 1.0

        self._params = PreprocessorParams(
            keep_band_mask=keep_mask,
            kept_wavelengths=kept_wl,
            apply_sg=self.apply_sg,
            sg_window=self.sg_window,
            sg_polyorder=self.sg_polyorder,
            sg_deriv=self.sg_deriv,
            apply_snv=self.apply_snv,
            mu_bands=mu_bands,
            sigma_bands=sigma_bands,
            mu_y=mu_y,
            sigma_y=sigma_y,
            n_safety_datasets=len(safety_datasets),
        )
        return self

    # transform --------------------------------------------------------------

    def transform(
        self, dataset: SpectralDataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplica phi (ajustado no treino) a qualquer dataset.

        Retorna (X_transformado, y_transformado).
        """
        params = self._require_fitted()

        if dataset.X.shape[1] != params.keep_band_mask.shape[0]:
            raise ValueError(
                f"Incompatibilidade de bandas: fit usou "
                f"{params.keep_band_mask.shape[0]} bandas, "
                f"transform recebeu {dataset.X.shape[1]}"
            )

        # 1. descarte de bandas
        X = dataset.X[:, params.keep_band_mask]

        # 2. SG (com os mesmos parâmetros do fit)
        if params.apply_sg:
            X = savgol_filter(
                X,
                window_length=params.sg_window,
                polyorder=params.sg_polyorder,
                deriv=params.sg_deriv,
                axis=1,
            )

        # 3. SNV
        if params.apply_snv:
            X = self._snv(X)

        # 4. Z-score com parâmetros do TREINO
        X = (X - params.mu_bands) / (params.sigma_bands + _EPS)

        # alvo
        y = (dataset.y - params.mu_y) / (params.sigma_y + _EPS)

        return X, y

    def fit_transform(
        self,
        dataset: SpectralDataset,
        safety_datasets: Sequence[SpectralDataset] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conveniência: fit no dataset e transform do mesmo dataset."""
        return self.fit(dataset, safety_datasets=safety_datasets).transform(
            dataset
        )

    # desnormalização do alvo ------------------------------------------------

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Recupera y na escala original (para cálculo de métricas)."""
        params = self._require_fitted()
        return y_norm * (params.sigma_y + _EPS) + params.mu_y

    # acesso aos parâmetros --------------------------------------------------

    @property
    def params(self) -> PreprocessorParams:
        """Os parâmetros ajustados (phi). Levanta se não ajustado."""
        return self._require_fitted()

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    # helpers internos -------------------------------------------------------

    def _compute_keep_mask(
        self,
        dataset: SpectralDataset,
        safety_datasets: Iterable[SpectralDataset] = (),
    ) -> np.ndarray:
        """Máscara True/False das bandas a manter."""
        p = dataset.X.shape[1]
        mask = np.ones(p, dtype=bool)
        if not self.drop_zero_bands:
            return mask

        all_zero_wl: list[float] = list(
            dataset.metadata.get("bands_systematically_zero", [])
        )
        for sd in safety_datasets:
            all_zero_wl.extend(sd.metadata.get("bands_systematically_zero", []))

        if not all_zero_wl:
            return mask

        zero_arr = np.asarray(all_zero_wl, dtype=np.float64)
        for j, wl in enumerate(dataset.wavelengths):
            if np.any(np.abs(zero_arr - wl) < _WL_MATCH_TOLERANCE_NM):
                mask[j] = False
        return mask

    def _apply_sg(self, X: np.ndarray) -> np.ndarray:
        """Aplica Savitzky-Golay ao longo do eixo espectral (axis=1).

        Pré-condição: os parâmetros já foram validados em __init__.
        """
        return savgol_filter(
            X,
            window_length=self.sg_window,
            polyorder=self.sg_polyorder,
            deriv=self.sg_deriv,
            axis=1,
        )

    @staticmethod
    def _snv(X: np.ndarray) -> np.ndarray:
        """Standard Normal Variate: padroniza cada linha (espectro)."""
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True)
        return (X - mu) / (sigma + _EPS)

    def _require_fitted(self) -> PreprocessorParams:
        if self._params is None:
            raise RuntimeError(
                "Preprocessor não foi ajustado. Chame .fit(dataset_treino) "
                "antes de .transform()."
            )
        return self._params
