# XAI para Reconhecimento de Emoções Faciais

Projeto de Explicabilidade (XAI) aplicada a modelos de deep learning para reconhecimento de emoções faciais, comparando Vision Transformers (ViT) e ConvNeXt (CNN).

## 📋 Objetivo da Pesquisa

Investigar **onde** os modelos focam ao classificar emoções faciais, comparando:
- **ViT**: Mecanismos de atenção (attention maps)
- **CNN**: Métodos baseados em gradiente (CAM)
- **Agnósticos**: LIME e SHAP (independentes de arquitetura)

## 🧠 Decisões de Implementação

### Por que Seleção Estratificada de Heatmaps?

Processar milhares de imagens gera GBs de heatmaps. A **seleção estratificada** resolve isso:

```
7 classes × 4 buckets × 5 imagens = 140 heatmaps representativos
```

**Os 4 buckets capturam cenários de pesquisa importantes:**
| Bucket | Descrição | Interesse |
|--------|-----------|-----------|
| `correct_high` | Acertou com confiança alta | Caso ideal |
| `correct_low` | Acertou com confiança baixa | Possível sorte |
| `wrong_high` | Errou com confiança alta | Caso problemático |
| `wrong_low` | Errou com confiança baixa | Imagem ambígua |

### Por que LIME/SHAP apenas nas imagens estratificadas?

- LIME/SHAP são **~10-100x mais lentos** que métodos nativos
- Objetivo: **comparar heatmaps** lado a lado (não calcular métricas)
- Usar os mesmos 140 casos permite comparação direta ViT vs CNN vs LIME/SHAP

### Por que LayerCAM ao invés de ScoreCAM?

ScoreCAM é mais robusto mas **~10x mais lento** (inviável para datasets grandes). LayerCAM oferece bom equilíbrio velocidade/qualidade.

## 🚀 Instalação

```bash
# Clone o repositório
git clone <https://github.com/rochatobias/XAI_fer.git>
cd IC-Projeto

# Crie e ative ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale dependências
pip install -r XAI/requirements.txt
```

## ⚙️ Configuração

### Onde configurar os caminhos

Edite `XAI/scripts/config.py`:

```python
# ============ MODELOS - MODIFIQUE AQUI SE NECESSÁRIO ============
VIT_MODEL_DIR = str(PROJECT_ROOT / "Training" / "Models" / "ViT" / "best_checkpoint-45153")
CNN_MODEL_PATH = str(PROJECT_ROOT / "Training" / "Models" / "CNN" / "convnext_fold_5_best.pth")

# ============ DATASET - MODIFIQUE AQUI SE NECESSÁRIO ============
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")
```

### Requisitos dos Modelos

> ⚠️ **IMPORTANTE**: Os modelos devem usar os mesmos parâmetros de pré-processamento do **ConvNeXt Base**:
> - Input size: 224×224
> - Mean: (0.485, 0.456, 0.406)
> - Std: (0.229, 0.224, 0.225)

Se treinou modelos com parâmetros diferentes, ajuste em `config.py`:
```python
IMG_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
```

### Estrutura esperada do dataset

```
XAI/data/aplicaçãoXAI/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

### Parâmetros principais

```python
N_SAMPLES = 1000         # Imagens para métricas (ViT/CNN)
N_SAMPLES_AGNOSTIC = 50  # Limite para LIME/SHAP (dentro do estratificado)
```

## 🎯 Como Executar

### Menu Interativo (Recomendado)

Execute sem argumentos para abrir o menu:
```bash
cd XAI/scripts
python main.py
```

```
════════════════════════════════════════════════════════════
         XAI ANALYSIS - MENU PRINCIPAL
════════════════════════════════════════════════════════════

  [1] Pipeline Completo (ViT + CNN)
  [2] Pipeline Apenas ViT
  [3] Pipeline Apenas CNN
  [4] Executar LIME/SHAP (imagens estratificadas)
  [5] Regenerar Gráficos (usa CSV existente)
  [6] Regenerar CSVs de Análise
  [0] Sair
```

O menu solicita interativamente a quantidade de imagens e outras opções.

---

### Via Linha de Comando (Flags)

Para automação ou scripts, use as flags:

#### Execução do Pipeline

```bash
# Pipeline completo (ViT + CNN) com 1000 imagens
python main.py --n_samples 1000

# Apenas ViT
python main.py --models vit --n_samples 500

# Apenas CNN
python main.py --models cnn --n_samples 500

# Com LIME/SHAP (10 imagens do estratificado)
python main.py --n_samples 1000 --agnostic --n_agnostic 10
```

#### Regenerar Outputs (sem reprocessar)

```bash
# Regenerar apenas gráficos
python main.py --plots-only

# Regenerar apenas CSVs de análise
python main.py --analysis-only

# Executar apenas LIME/SHAP (usa CSV existente)
python main.py --agnostic-only
```

#### Flags de Controle

| Flag | Alias | Descrição |
|------|-------|-----------|
| `--n_samples N` | `-n` | Número de imagens para análise |
| `--n_agnostic N` | | Número de imagens para LIME/SHAP |
| `--models [vit/cnn]` | `-m` | Modelos a processar |
| `--agnostic` | `-a` | Executar LIME/SHAP nas imagens estratificadas |
| `--agnostic-only` | | Apenas LIME/SHAP (usa CSV existente) |
| `--plots-only` | | Regenerar apenas gráficos |
| `--analysis-only` | | Regenerar apenas CSVs de análise |
| `--no-heatmaps` | | Não salvar visualizações |
| `--no-plots` | | Não gerar gráficos de resumo |
| `--no-analysis` | | Não gerar CSVs de análise |
| `--quiet` | `-q` | Modo silencioso |

#### Exemplos Combinados

```bash
# Pipeline rápido para teste (10 imagens, só ViT, sem plots)
python main.py -n 10 -m vit --no-plots --no-analysis

# Pipeline completo com LIME/SHAP customizado
python main.py --n_samples 5000 --agnostic --n_agnostic 50

# Modo silencioso para scripts
python main.py --n_samples 1000 --quiet
```

---


## 📊 Métricas Calculadas

### Fidelidade (quão bem o heatmap explica a decisão)
| Métrica | Descrição |
|---------|-----------|
| **AOPC** | Queda de confiança ao remover regiões importantes |
| **Insertion AUC** | Confiança ao adicionar pixels importantes |
| **Deletion AUC** | Confiança ao remover pixels importantes |

### Localidade (concentração do heatmap)
| Métrica | Descrição |
|---------|-----------|
| **Area@50/90** | % de área para capturar 50%/90% da atenção |
| **Gini** | Coeficiente de concentração (maior = mais focado) |
| **Entropy** | Dispersão (menor = mais focado) |

## 📁 Estrutura do Projeto

```
IC-Projeto/
├── Training/
│   └── Models/
│       ├── ViT/              # Seu checkpoint ViT
│       └── CNN/              # Seu peso ConvNeXt
├── XAI/
│   ├── data/
│   │   └── aplicaçãoXAI/     # Dataset (7 pastas de emoções)
│   ├── results/
│   │   ├── heatmaps/
│   │   │   ├── vit/          # Heatmaps ViT
│   │   │   ├── cnn/          # Heatmaps CNN
│   │   │   └── agnostic/     # Heatmaps LIME/SHAP
│   │   ├── summary/          # Gráficos de resumo
│   │   └── analysis/         # CSVs de análise
│   ├── scripts/
│   │   ├── config.py         # ⚙️ Configurações (modifique aqui)
│   │   ├── main.py           # Pipeline principal
│   │   ├── vit.py            # XAI para ViT
│   │   ├── cnn.py            # XAI para CNN
│   │   ├── agnostic.py       # LIME e SHAP
│   │   └── ...
│   ├── experiments.ipynb     # Notebook para testes
│   └── requirements.txt
└── README.md
```

##  Resultados Gerados

| Arquivo | Descrição |
|---------|-----------|
| `metrics_combined.csv` | Todas as métricas por imagem/método |
| `heatmap_selection.csv` | Imagens selecionadas e motivo |
| `heatmaps/vit/*.png` | Visualizações ViT |
| `heatmaps/cnn/*.png` | Visualizações CNN |
| `heatmaps/agnostic/*.png` | Visualizações LIME/SHAP |
| `summary/*.png` | Gráficos comparativos |
| `analysis/*.csv` | Estatísticas por método/classe |

## 📦 Dependências

- PyTorch ≥ 2.0
- Transformers (HuggingFace)
- timm
- grad-cam
- lime, shap (para métodos agnósticos)

## 👤 Autor

Tobias Rocha