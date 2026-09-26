"""pgsg_1 — Escalabilidade do Prior-Guided Spectral Gating em NIR.

Este pacote organiza a solução do projeto 1 (de 5) como uma composição de
oito operadores, espelhando a especificação formal em
docs/papers/pipeline_proposta.pdf:

    Pi = Sigma . I_interp . I_pred . M . C . P . T . G

onde cada operador é um submódulo deste pacote.

Submódulos
----------
ingestion         operador G — carrega (X, y, lambda) das bases
preprocessing     operador T — limpeza, espalhamento, normalização
priors            operador P — priors quimiométricos (ANOVA, VIP, RF)
scenarios         operador C — subamostragem controlada em n
models            operador M — PLS, MLP, CNN, PGSG, baselines
evaluation        operador I_pred — métricas com IC, testes pareados
interpretability  operador I_interp — coerência, estabilidade, atribuição
synthesis         operador Sigma — curvas, n*, n_p, mapas espectrais
"""

__version__ = "0.1.0"
