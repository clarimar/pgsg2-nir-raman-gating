"""preprocessing — operador T do pipeline.

Contrato (do documento de pipeline, seção 4)
--------------------------------------------
T retorna (X', y', phi), onde phi encapsula TODOS os parâmetros de
ajuste (bandas mantidas, coeficientes SG, mu/sigma de bandas e alvo).
phi é estimado EXCLUSIVAMENTE no treino e reaplicado ao teste —
Princípio da Higiene de Vazamento.

Implementação: classe `Preprocessor` no padrão fit/transform do sklearn.
Os parâmetros phi são encapsulados em `PreprocessorParams` (imutável).

Sub-transformações (ordem fixa — decisão de design)
---------------------------------------------------
1. Remoção de bandas zeradas       automática via metadata, opção de desligar — IMPLEMENTADO
2. Savitzky-Golay                  SG_{w,d,m}(X) — janela, ordem, derivada — IMPLEMENTADO
3. SNV (Standard Normal Variate)   por linha; corrige espalhamento — IMPLEMENTADO
4. Z-score por banda               (x - mu) / sigma; mu, sigma do treino — IMPLEMENTADO

O alvo y também é Z-score-normalizado; os parâmetros (mu_y, sigma_y) ficam
em phi para desnormalização no estágio de avaliação (inverse_transform_y).

Sobre o Savitzky-Golay
----------------------
SG é OPCIONAL e desligado por padrão (apply_sg=False). Quando ativado,
os três parâmetros (sg_window, sg_polyorder, sg_deriv) são obrigatórios
— sem defaults — para forçar escolha consciente por experimento. Anderson
et al. (2020) na Mango usaram (w=17, polyorder=2, deriv=2).

Sobre bandas zeradas variáveis por safra (ADR-0005)
---------------------------------------------------
O `fit` aceita `safety_datasets` opcionais: a máscara de descarte
resulta da UNIÃO das bandas zeradas entre treino e safety_datasets
(interseção das faixas úteis). Garantia da Comparabilidade entre treino
e teste em bases multi-instrumento.

API pública
-----------
- Preprocessor:        operador T (fit/transform)
- PreprocessorParams:  parâmetros ajustados (phi, imutável)

Invariantes (testados em tests/test_preprocessing.py)
-----------------------------------------------------
I1. phi ajustado no treino → mu, sigma calculados sem ver o teste
I2. T(X_teste, phi) usa exatamente os mesmos parâmetros que T(X_treino)
I3. Z-score: a média de X' no treino é ~0 e o desvio ~1
I4. SNV: a média de cada espectro pós-SNV é ~0 (por linha)
I5. Com safety_datasets, a faixa útil é a interseção (união dos zeros)
I6. SG é determinístico: mesmos parâmetros → mesmo resultado
"""

from .preprocessor import Preprocessor, PreprocessorParams

__all__ = ["Preprocessor", "PreprocessorParams"]
