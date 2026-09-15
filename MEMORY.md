# MEMORY.md — CTGAN & RDT Knowledge Base

> Este arquivo contém todo o conhecimento adquirido sobre os projetos CTGAN e RDT.
> Serve como memória persistente para modelos de LLM que atuarão nestes repositórios.

---

## 1. Visão Geral dos Projetos

### CTGAN (Conditional Tabular GAN)
- **Repositório**: https://github.com/sdv-dev/CTGAN
- **Fork do usuário**: https://github.com/maxsonferovante/CTGAN
- **Linguagem**: Python 3.11–3.13, PyTorch, Pandas
- **Propósito**: Geração de dados tabulares sintéticos usando GAN condicional
- **Artigo**: "Modeling Tabular data using Conditional GAN" — Lei Xu et al. (arXiv:1907.00503v2)

### RDT (Reversible Data Transforms)
- **Repositório**: https://github.com/sdv-dev/RDT
- **Fork do usuário**: https://github.com/maxsonferovante/RDT
- **Linguagem**: Python 3.11–3.13, NumPy, Pandas, scikit-learn
- **Propósito**: Transformação reversível de dados tabulares (normalização, encoding)
- **Uso no CTGAN**: ClusterBasedNormalizer + OneHotEncoder para pré-processamento

---

## 2. Arquitetura do CTGAN

### Pipeline Principal
```
Dados Tabulares → RDT (fit/transform) → CTGAN.fit(transformed_data) → CTGAN.sample(n) → RDT.inverse_transform(synthetic)
```

### Componentes Principais
- `ctgan/synthesizers/ctgan.py` — Classe CTGAN (treinamento GAN)
- `ctgan/synthesizers/tvae.py` — Classe TVAE (Variational Autoencoder)
- `ctgan/data_transformer.py` — DataTransformer (usa RDT fork diretamente)
- `ctgan/data_sampler.py` — DataSampler (amostragem condicional)
- `ctgan/condicional_dataset.py` — CondicionalDataset + CondicionalSampler (otimização)
- `ctgan/transformers.py` — Wrappers de transformers

### Fluxo de Treinamento
1. **DataTransformer.fit(data, discrete_columns)** — Ajusta RDT em cada coluna
2. **DataTransformer.transform(data)** — Transforma dados para formato numérico
3. **DataSampler.sample(num_samples)** — Gera batches com condicionamento
4. **CTGAN._train_epoch()** — Treina Discriminador e Gerador
5. **CTGAN.sample()** — Gera dados sintéticos
6. **DataTransformer.inverse_transform()** — Reverte para formato original

### RDT Usage no CTGAN
O CTGAN usa **diretamente o fork RDT** (não via pip install):
```python
# ctgan/data_transformer.py
from rdt.transformers.numerical import ClusterBasedNormalizer
from rdt.transformers.categorical import OneHotEncoder
```

### Colunas Discretas vs Contínuas
- **Discretas** (categóricas): Usam `OneHotEncoder` → outputs one-hot encoded
- **Contínuas** (numéricas): Usam `ClusterBasedNormalizer` → normaliza com mixture model

---

## 3. Arquitetura do RDT

### Componentes Principais
- `rdt/hyper_transformer.py` — HyperTransformer (orquestra transforms)
- `rdt/transformers/base.py` — BaseTransformer (classe base)
- `rdt/transformers/numerical.py` — FloatFormatter, ClusterBasedNormalizer, etc.
- `rdt/transformers/categorical.py` — FrequencyEncoder, OneHotEncoder, UniformEncoder, etc.
- `rdt/transformers/null.py` — NullTransformer

### ClusterBasedNormalizer (CBN)
- Usa `sklearn.mixture.BayesianGaussianMixture` para agrupar dados
- Normaliza cada cluster com z-score: `(x - mean) / (std * 4)`
- Seleciona componente via sampling estocástico
- **É o transformer mais usado em pipelines CTGAN**

### OneHotEncoder
- Mapeia categorias para one-hot vectors
- Lida com categorias não vistas durante fit
- Output: múltiplas colunas binárias

### HyperTransformer
- Detecta automaticamente tipos de colunas
- Aplica transformers apropriados por coluna
- Gerencia fit/transform/reverse_transform para todas as colunas

---

## 4. Branches Importantes

### CTGAN
| Branch | Descrição |
|--------|-----------|
| `main` | Código original (Pandas) — commits 66e5686, 355ea69 |
| `main-local` | Pandas + CondicionalDataset optimization |
| `feat/polars-migration` | Migração completa para Polars + sklearn |

### RDT
| Branch | Descrição |
|--------|-----------|
| `main` | Código original (Pandas) |
| `main-local` | Pandas + 3 otimizações de performance |

---

## 5. Migração CTGAN: Pandas → Polars

### Decisões de Migração
1. **sklearn para BayesianGMM** — substitui copulas/BayesianGaussianMixture
2. **Polars para tudo mais** — DataFrame I/O, operações
3. **Branch única**: `feat/polars-migration`
4. **Breaking change** — Polars DataFrame I/O (sem backward compat)
5. **Versão**: bump para 1.0.0
6. **Testes**: reescrever do zero

### Status da Migração
- **80 tests passando** (Polars branch)
- **66 tests passando** (Pandas main branch)
- Todos os arquivos migrados e commitados

### RDT Usage na Migração
O CTGAN Polars usa o RDT fork diretamente:
```python
# ctgan/data_transformer.py
from rdt.transformers.numerical import ClusterBasedNormalizer
from rdt.transformers.categorical import OneHotEncoder
```

---

## 6. Otimizações de Performance (RDT)

### Otimização 1: data.copy()
- **Arquivo**: `rdt/transformers/base.py:422,475`
- **Problema**: O(n) cópias de DataFrame a cada transformer
- **Solução**: Remover `data.copy()` — `drop()` já retorna novo DataFrame
- **Resultado**: -8.5% no tempo total do HyperTransformer

### Otimização 2: apply()
- **Arquivo**: `rdt/transformers/categorical.py:551-553`
- **Problema**: `data.apply(self._get_value)` — O(n) chamadas Python
- **Solução**: Vectorização com mask booleano numpy
- **Resultado**: Reduz O(n) para O(k) iterações (k = número de categorias)

### Otimização 3: Loop Python
- **Arquivo**: `rdt/transformers/numerical.py:615-622`
- **Problema**: Loop puro `for i in range(len(data))` com `np.random.choice`
- **Solução**: `argmax(rand_vals[:, None] <= cumsum)` vectorizado
- **Resultado**: ClusterBasedNormalizer Transform: 284.4ms → 13.1ms (-95.4%, 21.7x mais rápido)

---

## 7. Resultados de Benchmark (50k rows × 20 cols)

### HyperTransformer
| Métrica | Original | Otimizado | Melhoria |
|---------|----------|-----------|----------|
| Fit | 269.4ms | 252.4ms | -6.3% |
| Transform | 252.2ms | 223.0ms | -11.6% |
| Reverse Transform | 47.2ms | 44.9ms | -4.9% |
| Total | 568.8ms | 520.3ms | -8.5% |

### ClusterBasedNormalizer (Isolado)
| Métrica | Original | Otimizado | Melhoria |
|---------|----------|-----------|----------|
| Fit | 1466.5ms | 1538.8ms | +4.9%* |
| Transform | 284.4ms | 13.1ms | -95.4% 🔥 |
| Reverse Transform | 3.5ms | 2.4ms | -31.4% |

### FrequencyEncoder (Isolado)
| Métrica | Original | Otimizado | Melhoria |
|---------|----------|-----------|----------|
| Fit | 13.1ms | 13.4ms | +2.3%* |
| Transform | 7.4ms | 7.1ms | -4.1% |
| Reverse Transform | 4.3ms | 3.3ms | -23.3% |

*\* Variações menores no Fit são devidas a ruído de medição*

### Memória
- Peak: 46.0MB (original) vs 53.7MB (otimizado) — variação de processo

---

## 8. Resultados de Benchmark CTGAN vs TVAE

### Pipeline Completo (50k rows, 20 cols, 5 epochs)
| Scenario | CTGAN | TVAE | CTGAN Memory |
|----------|-------|------|-------------|
| Pandas Normal | 229.1s | 88.9s | 308.1 MB |
| Polars Normal | 77.1s | 57.5s | 164.4 MB |
| Pandas Optimized | **52.5s** | **32.8s** | 246.4 MB |
| Polars Optimized | 75.1s | 55.6s | **164.4 MB** |

---

## 9. Documentação (HTML)

### Páginas Criadas
| Arquivo | Descrição |
|---------|-----------|
| `ctgan-documentation.html` | Página principal com links |
| `ctgan-simulation.html` | Simulação detalhada com contexto bancário |
| `ctgan-rdt.html` | RDT Deep Dive (13 seções) |
| `ctgan-normalizacao-rdt.html` | Processo de normalização |
| `ctgan-rdt-uso.html` | CTGAN ↔ RDT análise |
| `ctgan-performance-ctgan-vs-tvae.html` | Performance CTGAN vs TVAE |
| `ctgan-tvae.html` | Documentação TVAE |
| `ctgan-comparacao.html` | Comparação CTGAN vs TVAE |
| `ctgan-privacidade.html` | Riscos de privacidade/fuga de dados |
| `ctgan-differential-privacy.html` | Implementação DP (dpsdg + Opacus) |
| `rdt-performance.html` | Otimizações de performance RDT |

### Repositório de Documentação
- https://github.com/maxsonferovante/ctgan-documentation

---

## 10. Privacidade e Differential Privacy

### Riscos de Fuga de Dados
- GANs podem memorizar e reproduzir dados reais
- Modelos generativos estão sujeitos a membership inference attacks
- Differential Privacy (DP) é a solução padrão

### Bibliotecas DP
- **dpsdg** (recomendado): https://github.com/brains-group/dpsdg — DP-CTGAN e DP-TVAE
- **Opacus** (alternativa): https://github.com/meta-pytorch/opacus — DP-SGD manual

### Referências Acadêmicas
- "When Privacy Isn't Synthetic" — https://academic.oup.com/bioinformatics/article/41/11/btaf527/8261371
- "The DCR Delusion" — https://link.springer.com/chapter/10.1007/978-3-032-07884-1_24
- "Preserving information while respecting privacy" — https://www.nature.com/articles/s41746-025-01431-6
- "On the Privacy Properties of GAN-generated Samples" — http://proceedings.mlr.press/v130/lin21b/lin21b.pdf
- "Risk In Context" — https://arxiv.org/abs/2507.17066

---

## 11. Ambiente de Desenvolvimento

### Ferramentas
- **Python**: 3.11–3.13 (ambiente local: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`)
- **Polars**: 1.44.2
- **Docker**: 29.7.2 + Compose v5.4.0
- **pytest**: 8.4.2

### Comandos Úteis
```bash
# RDT — executar testes
cd rdt && python3 -m pytest tests/unit/ -q
cd rdt && python3 -m pytest tests/integration/test_hyper_transformer.py -q

# CTGAN — executar testes
cd CTGAN && python3 -m pytest tests/ -q

# Benchmark RDT
cd rdt && python3 -c "import time, numpy as np, pandas as pd; from rdt import HyperTransformer; ..."

# Push branches
git remote add myfork https://github.com/maxsonferovante/CTGAN.git
git push myfork main-local
```

### Estrutura de Testes
- `tests/unit/` — 674 testes unitários (RDT)
- `tests/integration/` — 160+ testes de integração (RDT)
- `tests/` — 80 testes (CTGAN Polars), 66 testes (CTGAN Pandas)

---

## 12. Conhecimento Técnico Chave

### Pandas vs Polars
- Polars é significativamente mais rápido para operações de DataFrame
- CTGAN Pandas Normal: 229.1s vs CTGAN Polars Normal: 77.1s (3x mais rápido)
- CTGAN Polars Optimized: 75.1s vs CTGAN Pandas Optimized: 52.5s

### CondicionalDataset Optimization
- Elimina ~1,500 conversões NumPy→Tensor por treinamento CTGAN
- Usa `how='horizontal_extend'` para polars concat
- Implementado em `ctgan/condicional_dataset.py`

### ClusterBasedNormalizer Internals
- Usa `BayesianGaussianMixture` do sklearn
- `valid_component_indicator` filtra componentes com peso > threshold
- Normalização: `(x - mean) / (STD_MULTIPLIER * std)` onde STD_MULTIPLIER = 4
- Sampling: argmax(rand_vals ≤ cumsum) vectorizado

### Import Structure (RDT)
```python
# Lazy imports para evitar importação circular
# rdt/__init__.py — importslazy
# rdt/transformers/__init__.py — importslazy

# Uso correto no CTGAN:
from rdt.transformers.numerical import ClusterBasedNormalizer
from rdt.transformers.categorical import OneHotEncoder
```

---

## 13. Bloqueios e Restrições

### Permissões
- **Não é possível push para sdv-dev/CTGAN** — erro 403 (permission denied)
- Forks criados em: maxsonferovante/CTGAN e maxsonferovante/RDT

### Compatibilidade
- Polars DataFrame I/O é breaking change (sem backward compat)
- Versão bump → 1.0.0 na migração Polars
- Testes reescritos do zero para Polars

---

## 14. Próximos Passos

### CTGAN
- [ ] Continuar otimizações no branch `feat/polars-migration`
- [ ] Testar pipeline completo com Docker benchmark
- [ ] Avaliar integração com DPSDG para Differential Privacy

### RDT
- [ ] Avaliar outras otimizações (type hints, duplicação de código)
- [ ] Testar com datasets maiores (100k+ rows)
- [ ] Considerar contribuição upstream para sdv-dev/RDT

### Documentação
- [ ] Atualizar páginas HTML com novos resultados
- [ ] Adicionar exemplos de código completos
- [ ] Documentar API do CTGAN para desenvolvedores

---

*Última atualização: 2026-09-14*
*Autor: Maxson Ferovante*
