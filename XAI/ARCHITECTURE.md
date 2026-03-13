# Arquitetura e Decisões do Projeto XAI-FER

Documento técnico explicando as decisões de arquitetura, ferramentas, métodos XAI, gráficos gerados e pontos de melhoria para a pesquisa.

---

## 1. Organização do Código

### Antes (v1 — scripts soltos)

14 arquivos `.py` na pasta `scripts/`, sem pacote Python. Imports relativos, funções de plot duplicadas em `vit.py` e `cnn.py`, acoplamento forte entre módulos.

### Depois (v2 — pacote `xai_fer/`)

```
xai_fer/
├── __init__.py           # Metadados do pacote
├── __main__.py           # Entry point (CLI + menu interativo)
├── config.py             # Configuração centralizada
├── utils.py              # Mapeamento de labels
├── data/loader.py        # Carregamento e amostragem
├── models/
│   ├── vit_explainer.py  # Métodos de atenção (Raw/Rollout/Flow)
│   ├── cnn_explainer.py  # Métodos CAM (GradCAM/GradCAM++/LayerCAM)
│   └── agnostic_explainer.py  # LIME e SHAP
├── evaluation/
│   ├── metrics.py        # Fidelidade (AOPC, Insertion/Deletion) e localidade
│   └── analysis.py       # Geração de CSVs analíticos e bootstrap
├── visualization/
│   ├── heatmaps.py       # Overlays individuais
│   └── plots.py          # Gráficos de sumário (6 tipos)
├── pipeline/
│   ├── runner.py          # Orquestração do loop de processamento
│   └── selector.py        # Seleção estratificada (7 classes × 4 buckets)
└── reporting/
    └── latex_table.py     # Tabela LaTeX para o paper
```

**Princípios aplicados:**
- Separação de responsabilidades (cada módulo faz uma coisa)
- Imports absolutos (`from xai_fer.config import ...`)
- Docstrings descritivas em todo arquivo e função
- Sem funções duplicadas (plots removidos de `vit.py` e `cnn.py`)

---

## 2. Decisões de Ferramentas

| Ferramenta | Uso | Justificativa |
|---|---|---|
| **PyTorch** | Modelos, tensores, GPU | Ecossistema dominante em pesquisa de DL |
| **HuggingFace Transformers** | ViT (checkpoint fine-tuned) | Suporte nativo a `output_attentions=True` |
| **timm** | ConvNeXt + pré-processamento | Maior variedade de modelos, `resolve_model_data_config` |
| **pytorch-grad-cam** | GradCAM, GradCAM++, LayerCAM | Biblioteca especializada, manutenção ativa |
| **lime** | LIME | Implementação de referência (Ribeiro et al.) |
| **shap** | SHAP (partition) | Implementação oficial, algoritmo partition viável |
| **scikit-image** | SLIC (superpixels para LIME) | Padrão para segmentação perceptual |
| **matplotlib** | Todos os gráficos | Flexibilidade total para customização |
| **pandas** | DataFrames, CSVs | Análise tabular e export |
| **NumPy** | Métricas de localidade | Cálculos vetorizados sem GPU |

---

## 3. Métodos XAI — Justificativas

### ViT (baseados em atenção)

| Método | Descrição | Por que incluir |
|---|---|---|
| **Raw Attention** | Atenção da última camada, média das heads | Baseline simples; rápido |
| **Attention Rollout** | Composição multiplicativa camada a camada | Captura influência acumulada do [CLS] |
| **Attention Flow** | Propagação de massa com exponenciação | Destaca conexões fortes (power > 1) |

### CNN (CAM-based)

| Método | Descrição | Por que incluir |
|---|---|---|
| **GradCAM** | Gradientes médios × ativações | Padrão da literatura, referência |
| **GradCAM++** | Pesos de 2ª ordem | Melhor com objetos múltiplos/parciais |
| **LayerCAM** | Gradientes positivos × ativações | Bom equilíbrio velocidade/qualidade |

> **ScoreCAM** foi descartado por ser ~10× mais lento sem ganho proporcional em nosso dataset.

### Agnósticos

| Método | Descrição | Por que incluir |
|---|---|---|
| **LIME** | Perturbação de superpixels + modelo linear local | Referência em XAI model-agnostic |
| **SHAP** | Valores de Shapley por partição | Fundamentação teórica sólida |

> Aplicados apenas ao subconjunto estratificado (~140 imagens) pois são 10-100× mais lentos.

---

## 4. Gráficos Gerados

### 4.1 `metrics_by_method.png`
**Barplot de métricas de fidelidade por método XAI.** Compara AOPC_mean, AOPC_zero, Insertion_AUC e Deletion_AUC para cada método. Permite ver qual método gera heatmaps mais fiéis à decisão do modelo.

### 4.2 `model_comparison.png`
**Comparação ViT vs CNN em 4 métricas-chave.** Mostra AOPC, Insertion AUC, Deletion AUC e Gini lado a lado. Identifica se uma arquitetura produz explicações superiores.

### 4.3 `metrics_radar.png`
**Métricas de localidade por método (barplots verticais).** Concentration AUC, Entropy e Area@50, com paleta fria (CNN) vs quente (ViT). Mostra concentração vs dispersão dos heatmaps.

### 4.4 `confidence_distribution.png`
**Histograma de confiança separado por acerto/erro.** Revela calibração do modelo — sobreposição grande = calibração ruim.

### 4.5 `accuracy_by_class.png`
**Acurácia por classe de emoção, por modelo.** Identifica classes problemáticas (ex: disgust, fear) e diferenças entre ViT e CNN.

### 4.6 `insertion_deletion_curves.png`
**Curvas médias de Insertion e Deletion (central no paper).** Mostra como a confiança responde à inserção/remoção progressiva de regiões. AUCs dessas curvas são as métricas numéricas.

---

## 5. Pontos de Melhoria para a Pesquisa

### Nível de Experimento

1. **Sample size**: `N_SAMPLES = 1` no config atual — para resultados publicáveis, usar no mínimo 500-1000 imagens por modelo.
2. **Reprodutibilidade**: fixar seed global de PyTorch (`torch.manual_seed`) além do `random.seed` já usado no loader.
3. **Validação cruzada das métricas**: calcular intervalos de confiança (bootstrap já implementado) e reportar no paper.
4. **Significância estatística**: implementar testes pareados (Wilcoxon signed-rank) para comparar métodos XAI, ao invés de apenas comparar médias.
5. **Cross-dataset**: testar em outros datasets de FER (AffectNet, RAF-DB) para generalização.
6. **Calibração**: adicionar métricas de calibração (ECE, reliability diagram) para quantificar o overconfidence observado nos histogramas.

### Nível de Implementação

7. **ScoreCAM como ablation**: incluir ao menos uma comparação com ScoreCAM em amostra reduzida, mesmo que lento.
8. **Métricas por classe no LIME/SHAP**: atualmente LIME/SHAP geram apenas heatmaps; implementar métricas de fidelidade também para eles, permitindo comparação quantitativa com métodos nativos.
9. **Normalização de heatmap SHAP**: o SHAP retorna escala simétrica [-1, 1] enquanto outros métodos retornam [0, 1] — isso dificulta comparação direta. Considerar normalizar todos para [0, 1] antes de calcular métricas.
10. **Caching de modelos**: `load_cnn_model()` é chamado duas vezes no pipeline completo — implementar cache de modelo para reduzir tempo e memória.
11. **GPU memory management**: limpar cache CUDA entre modelos (`torch.cuda.empty_cache()`) para evitar OOM em GPUs com pouca memória.
12. **Logging estruturado**: migrar de `print()` para `logging` com níveis (DEBUG/INFO/WARNING) para facilitar debug e análise pós-execução.

### Nível de Paper

13. **Comparação com ground truth**: se disponível, usar eye-tracking ou anotações de AUs (Action Units) como ground truth para validar os heatmaps.
14. **User study**: avaliar qualitativamente com humanos qual método XAI produz explicações mais interpretáveis.
15. **Análise de failure modes**: expandir a análise de `high_conf_errors` com estudo qualitativo dos heatmaps nessas imagens.
