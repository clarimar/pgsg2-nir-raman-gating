"""
Operador I_interp (interpretabilidade) de pgsg_2: formalização dos três
descritores de "topologia do gate espectral" (Seção "Topologia do gate
espectral" de pgsg_2_proposta.tex):

    - Entropia de Shannon: H(g) = -sum(g_i log g_i)
    - Suavidade (variação total normalizada): TV(g) = mean(|g_i - g_{i+1}|)
    - Esparsidade (índice de Hoyer): S(g) em [0,1]

Nota de interpretação (importante, ver ADR/achado de H1): como o
PGSGModel de pgsg_1 parou em best_epoch=0 no experimento Raman (o gate
não se afasta do prior de inicialização), estes descritores aplicados
ao "gate aprendido" nesse experimento caracterizam, na prática, a
topologia do PRIOR de inicialização, não de um gate genuinamente
otimizado. As funções abaixo são agnósticas a essa distinção -- apenas
recebem um vetor g e descrevem sua estrutura -- mas a interpretação
científica dos números deve declarar explicitamente qual dos dois casos
se aplica.
"""

from __future__ import annotations

import warnings

import numpy as np

_EPS = 1e-300  # evita log(0) exato sem afetar o valor numérico da entropia


def gate_entropy(g: np.ndarray, *, renormalize_if_needed: bool = True) -> float:
    """Entropia de Shannon do gate.

    Pressupõe g_i > 0 e sum(g) == 1 (verdadeiro por construção para o
    gate softmax do PGSGModel). Se sum(g) != 1 (ex.: um gate de outra
    arquitetura, não normalizado), renormaliza com um aviso -- a menos
    que renormalize_if_needed=False, caso em que levanta.
    """
    g = np.asarray(g, dtype=float)
    _validate_gate_nonnegative(g)

    total = g.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        if not renormalize_if_needed:
            raise ValueError(
                f"gate_entropy: soma(g)={total:.6f} != 1 e renormalize_if_needed=False"
            )
        warnings.warn(
            f"gate_entropy: soma(g)={total:.6f} != 1; renormalizando antes de calcular H(g). "
            "Isso é esperado para gates que não vêm de softmax.",
            stacklevel=2,
        )
        g = g / total

    return float(-np.sum(g * np.log(g + _EPS)))


def gate_smoothness(g: np.ndarray) -> float:
    """Suavidade (variação total normalizada) do gate: TV(g) = mean(|g_i - g_{i+1}|).

    Valores baixos indicam gate suave (bandas vizinhas com peso
    parecido); valores altos indicam variação abrupta entre bandas
    adjacentes.
    """
    g = np.asarray(g, dtype=float)
    _validate_gate_nonnegative(g)
    if g.shape[0] < 2:
        raise ValueError("gate_smoothness requer ao menos 2 bandas")
    return float(np.mean(np.abs(np.diff(g))))


def gate_sparsity_hoyer(g: np.ndarray) -> float:
    """Índice de Hoyer de esparsidade: S(g) = (sqrt(p) - ||g||_1/||g||_2) / (sqrt(p) - 1).

    S(g)=0 para gate uniformemente denso; S(g)=1 para gate one-hot.
    """
    g = np.asarray(g, dtype=float)
    _validate_gate_nonnegative(g)
    p = g.shape[0]
    if p < 2:
        raise ValueError("gate_sparsity_hoyer requer ao menos 2 bandas")

    l1 = np.sum(np.abs(g))
    l2 = np.sqrt(np.sum(g ** 2))
    if l2 < 1e-300:
        # gate todo-zero (degenerado); tratado como caso limite denso.
        return 0.0

    sqrt_p = np.sqrt(p)
    return float((sqrt_p - l1 / l2) / (sqrt_p - 1))


def gate_topology(g: np.ndarray) -> dict[str, float]:
    """Calcula os três descritores de uma vez, para conveniência."""
    return {
        "entropy": gate_entropy(g),
        "smoothness": gate_smoothness(g),
        "sparsity_hoyer": gate_sparsity_hoyer(g),
    }


def _validate_gate_nonnegative(g: np.ndarray) -> None:
    if g.ndim != 1:
        raise ValueError(f"gate deve ser 1D, recebido {g.ndim}D")
    if np.any(g < -1e-9):
        raise ValueError("gate contém valores negativos (fora do intervalo [0,1])")
    if np.isnan(g).any():
        raise ValueError("gate contém NaN")
