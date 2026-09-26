"""Classes base do operador M (modelagem).

Interface uniforme para todos os modelos do projeto (PLS, MLP, CNN, PGSG),
seguindo o padrão fit/predict do scikit-learn. A uniformidade permite que
o estágio I_pred (avaliação) itere sobre modelos sem ramificações.

Duas classes:
- Model: interface base para qualquer modelo de regressão espectral.
- GatedModel: subclasse para modelos com camada de gating (PGSG e variantes),
  expõe `gates` e `prior_used` para o estágio I_interp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pgsg_1.ingestion import SpectralDataset


class Model(ABC):
    """Interface base para todos os modelos do operador M.

    Subclasses implementam `_fit_impl` e `_predict_impl`. A classe base
    cuida de:
    - validação de entrada (treino é SpectralDataset; teste idem)
    - flag `is_fitted`
    - mensagens de erro consistentes

    Contrato
    --------
    - `fit(train, prior=None)`: ajusta o modelo APENAS no treino.
      O parâmetro `prior` é ignorado por modelos não-gated.
    - `predict(dataset)`: devolve y_hat na escala ORIGINAL do alvo.
      Note que o pipeline normaliza y no Preprocessor, mas o Model
      assume que `train.y` já é o alvo na escala que o experimento
      espera ver de volta. A des-normalização é responsabilidade do
      consumidor (estágio I_pred), não do Model.

    Higiene de vazamento
    --------------------
    O `predict(test)` NUNCA pode disparar `fit`. Reaplicação a um conjunto
    de teste é puramente uma operação de inferência.
    """

    def __init__(self) -> None:
        self._fitted = False

    @property
    def name(self) -> str:
        """Identificador curto. Por padrão, o nome da classe sem 'Model'."""
        return self.__class__.__name__.replace("Model", "").upper()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        train: SpectralDataset,
        prior: np.ndarray | None = None,
    ) -> "Model":
        """Ajusta o modelo no treino.

        Parâmetros
        ----------
        train : SpectralDataset
            Dataset de treino. APENAS este é usado para ajuste.
        prior : np.ndarray | None
            Vetor s em [0,1]^p para inicialização do gating (só usado por
            GatedModel). Modelos não-gated ignoram este parâmetro.

        Retorna
        -------
        self, para permitir encadeamento `Model().fit(train).predict(test)`.
        """
        _validate_train(train)
        if prior is not None:
            _validate_prior(prior, train.n_bands, self.name)

        self._fit_impl(train, prior)
        self._fitted = True
        return self

    def predict(self, dataset: SpectralDataset) -> np.ndarray:
        """Devolve y_hat sobre `dataset` (treino ou teste).

        Retorna
        -------
        np.ndarray de shape (dataset.n_samples,), na escala ORIGINAL de y.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} não foi ajustado. Chame .fit(train) antes."
            )
        y_hat = self._predict_impl(dataset)
        # validação defensiva: subclasse deve devolver shape correto
        if y_hat.shape != (dataset.n_samples,):
            raise RuntimeError(
                f"{self.name}._predict_impl devolveu shape {y_hat.shape}, "
                f"esperado ({dataset.n_samples},)"
            )
        return y_hat

    # subclasses implementam estes dois métodos -----------------------------

    @abstractmethod
    def _fit_impl(
        self, train: SpectralDataset, prior: np.ndarray | None
    ) -> None:
        """Ajuste específico do modelo. Não retorna nada; usa atributos
        privados do self para guardar o estado."""

    @abstractmethod
    def _predict_impl(self, dataset: SpectralDataset) -> np.ndarray:
        """Predição específica do modelo. Retorna y_hat na escala original."""


class GatedModel(Model):
    """Modelos com camada de gating espectral (PGSG e variantes).

    Estende Model com duas propriedades adicionais para o estágio I_interp:
    - `gates`: o vetor g em [0,1]^p aprendido (soma 1, saída de softmax).
    - `prior_used`: o vetor s que foi usado para inicializar theta.

    Subclasses implementam `_gates_impl` e `_prior_used_impl`.
    """

    @property
    def gates(self) -> np.ndarray:
        """Vetor g em [0,1]^p após o treino."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} não foi ajustado. Chame .fit(train) antes."
            )
        g = self._gates_impl()
        # validações defensivas
        if g.ndim != 1:
            raise RuntimeError(
                f"{self.name}._gates_impl devolveu ndim={g.ndim}, esperado 1"
            )
        if not (g.min() >= -1e-9 and g.max() <= 1.0 + 1e-9):
            raise RuntimeError(
                f"{self.name}._gates_impl: gates fora de [0,1], "
                f"min={g.min()}, max={g.max()}"
            )
        return g

    @property
    def prior_used(self) -> np.ndarray:
        """O vetor de prior s que inicializou theta (None se random init)."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} não foi ajustado. Chame .fit(train) antes."
            )
        return self._prior_used_impl()

    @abstractmethod
    def _gates_impl(self) -> np.ndarray: ...

    @abstractmethod
    def _prior_used_impl(self) -> np.ndarray: ...


# helpers de validação -------------------------------------------------------


def _validate_train(train: SpectralDataset) -> None:
    """Validações comuns a todos os modelos."""
    if train.n_samples < 2:
        raise ValueError(
            f"Treino com {train.n_samples} amostras é insuficiente; "
            f"é necessário n >= 2"
        )


def _validate_prior(prior: np.ndarray, n_bands: int, model_name: str) -> None:
    """Validações do prior (quando passado)."""
    if prior.ndim != 1:
        raise ValueError(
            f"{model_name}: prior deve ser 1D, recebido {prior.ndim}D"
        )
    if prior.shape[0] != n_bands:
        raise ValueError(
            f"{model_name}: prior.shape[0]={prior.shape[0]} != "
            f"n_bands={n_bands}"
        )
    if prior.min() < 0 or prior.max() > 1:
        raise ValueError(
            f"{model_name}: prior deve estar em [0,1]; "
            f"recebido [{prior.min()}, {prior.max()}]"
        )
