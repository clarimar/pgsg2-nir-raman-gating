"""PLSModel — baseline quimiométrico padrão-ouro.

Wrapper fino sobre sklearn.cross_decomposition.PLSRegression.

PLS é o baseline contra o qual a curva Delta_PLS(n) = R²_PLS - R²_PGSG é
definida (hipótese H1 do projeto). Implementação simples: PLS não tem
treino iterativo, não tem gating, não tem hiperparâmetros além de
n_components.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from pgsg_1.ingestion import SpectralDataset

from .base import Model


class PLSModel(Model):
    """Partial Least Squares Regression.

    Parâmetros
    ----------
    n_components : int
        Número H de componentes (variáveis latentes). Default 10
        (convenção de Anderson 2020 e do artigo fundador).

    Observações
    -----------
    Para n_amostras < n_components, o sklearn levanta. Subimos com
    mensagem mais clara no _fit_impl.
    """

    def __init__(self, n_components: int = 10) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError(
                f"n_components deve ser >= 1; recebido {n_components}"
            )
        self.n_components = int(n_components)
        self._pls: PLSRegression | None = None
        self._effective_n_components: int | None = None

    @property
    def effective_n_components(self) -> int:
        """H efetivamente usado no fit. Pode ser < n_components se n pequeno."""
        if self._effective_n_components is None:
            raise RuntimeError(
                f"{self.name} não foi ajustado. Chame .fit(train) antes."
            )
        return self._effective_n_components

    def _fit_impl(
        self, train: SpectralDataset, prior: np.ndarray | None
    ) -> None:
        # PLS ignora prior (não é gated)
        n, p = train.X.shape
        # PLS exige n_components <= min(n_samples, n_features)
        # (na verdade < n_samples para evitar problemas numéricos)
        H = min(self.n_components, n - 1, p)
        H = max(H, 1)
        self._effective_n_components = H

        self._pls = PLSRegression(n_components=H, scale=True)
        self._pls.fit(train.X, train.y)

    def _predict_impl(self, dataset: SpectralDataset) -> np.ndarray:
        assert self._pls is not None
        y_hat = self._pls.predict(dataset.X)
        # sklearn devolve (n, 1) para y univariado; achatamos para (n,)
        return np.asarray(y_hat).ravel()
