# ADR-0001 (pgsg_2): Escolha do dataset Raman

## Status
Aceito

## Contexto
`pgsg_2` exige uma base de dados Raman pública, com alvo de regressão único e
quimicamente interpretável, e tamanho amostral suficiente para não reintroduzir
o problema de pequena amostra que `pgsg_1` já corrigiu no ramo NIR. A fonte
primária é o benchmark RamanBench, acessado via pacote `raman-data` (Koddenbrock
et al., 2026, arXiv:2605.02003), que padroniza 74 datasets públicos (58 de
regressão).

## Levantamento (via `raman_data.datasets.get_dataset_info`, sem download de dados)

| Dataset | N (declarado) | Alvo | Natureza | Fonte |
|---|---|---|---|---|
| `bioprocess_substrates` | 6.960 espectros | 8 metabólitos (glicose, glicerol, acetato, etc.) | real, medido | Lange et al. 2026, *Measurement* |
| `sugar_mixtures_low_snr` / `_high_snr` | 7.680 medições cada | concentração de açúcares em mistura | real, medido | Georgiev et al. 2024, PNAS |
| `chembl_molecules` | 140.000 espectros | propriedades quânticas (HOMO-LUMO, polarizabilidade, etc.) | **simulado (DFT)** | Liang et al. 2025, *Scientific Data* |
| `acetic_acid_species` e demais 5 ácidos (Echtermeyer) | não declarado em metadata; tipicamente pequeno (experimento de titulação) | concentração de espécie ácida dissociada | real, medido | Echtermeyer et al. 2021, *Applied Spectroscopy* |
| `bioprocess_analytes_{anton_532,anton_785,kaiser,metrohm,mettler_toledo,tec5,timegate,tornado}` | não declarado ("alguns têm mais amostras que outros") | glicose / acetato / sulfato de Mg | real, medido, 8 instrumentos sobre as mesmas amostras | Lange et al. 2025, *Spectrochimica Acta A* |
| `fuel_benchtop` / `fuel_handheld` | 179 amostras cada (pareadas) | RON, MON | real, medido, 2 instrumentos | Voigt 2019 / Legner 2019 |

## Decisão
Adotar **`bioprocess_substrates`**, com **glicose** como alvo único de regressão.

## Justificativa
1. **Escala real.** Maior N entre os datasets de medição real (6.960), evitando
   reabrir o problema de pequena amostra corrigido em `pgsg_1`. `chembl_molecules`
   tem N maior, mas é dado simulado por DFT — não comparável ao regime de
   medição física do Mango DMC v3.
2. **Alvo único, quimicamente interpretável.** Glicose é escolhida entre os 8
   metabólitos disponíveis para manter `pgsg_2` fiel ao seu escopo (generalização
   de *modalidade*). Regressão multi-alvo permanece reservada para `pgsg_3`.
3. **Robustez como benchmark.** O dataset foi desenhado explicitamente para
   testar robustez de regressão contra correlações espúrias de matriz (sais
   minerais, antifoam) — útil para uma discussão mais rigorosa de generalização
   no artigo.
4. **Paralelo temático com o NIR.** Monitoramento de bioprocesso (atributo de
   processo/qualidade via espectroscopia) é conceitualmente próximo ao
   monitoramento de qualidade do Mango DMC, facilitando a narrativa comparativa
   NIR↔Raman.

## Consequências
- O operador $\mathcal{G}$ (ingestão) precisa de um novo carregador para o
  `hf_key` `chlange/SubstrateMixRaman` (Hugging Face), com `metadata[domain] =
  "raman"` e unidade de eixo espectral em cm$^{-1}$.
- O download efetivo dos dados (`raman_data.datasets.load_dataset`) deve ser
  executado na máquina com GPU/60 GB RAM — o ambiente usado para este
  levantamento não tem acesso de rede a Hugging Face/Zenodo/Figshare (só
  PyPI/GitHub), então apenas os metadados foram inspecionados aqui.
- Os demais candidatos (`sugar_mixtures_*`, `bioprocess_analytes_*`,
  `fuel_benchtop`/`fuel_handheld`) ficam registrados como candidatos de
  robustez/generalização secundária, não bloqueantes para o núcleo empírico
  de `pgsg_2`.

## Referência
Lange, C., Altmann, M., Stors, D., Seidel, S., Moynahan, K., Cai, L., Born, S.,
Neubauer, P., Cruz Bournazou, M.N. *Deep learning for Raman spectroscopy:
Benchmarking models for upstream bioprocess monitoring*. Measurement, 258,
118884, 2026. DOI: 10.1016/j.measurement.2025.118884.
