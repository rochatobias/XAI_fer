# XAI para Reconhecimento de Emoções Faciais

Projeto de Explicabilidade (XAI) aplicada a modelos de deep learning para reconhecimento de emoções faciais, comparando Vision Transformers (ViT) e ConvNeXt (CNN).

## 📋 Objetivo da Pesquisa

Investigar **onde** os modelos focam ao classificar emoções faciais, comparando:
- **ViT**: Mecanismos de atenção (attention maps)
- **CNN**: Métodos baseados em gradiente (CAM)
- **Agnósticos**: LIME e SHAP (independentes de arquitetura)

> Para decisões detalhadas de arquitetura, ferramentas e pontos de melhoria, veja [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 🚀 Instalação

```bash
# Clone o repositório
git clone <https://github.com/rochatobias/XAI_fer.git>
cd IC-Projeto

# Crie e ative ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Instale dependências
pip install -r XAI/requirements.txt
```

## ⚙️ Configuração

Edite `XAI/xai_fer/config.py`:

```python
# ============ MODELOS ============
VIT_MODEL_DIR = str(PROJECT_ROOT / "Training" / "Models" / "ViT" / "best_checkpoint-45153")
CNN_MODEL_PATH = str(PROJECT_ROOT / "Training" / "Models" / "CNN" / "convnext_fold_5_best.pth")

# ============ DATASET ============
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")

# ============ AMOSTRAS ============
N_SAMPLES = 1000         # Imagens para métricas (ViT/CNN)
N_SAMPLES_AGNOSTIC = 50  # Limite para LIME/SHAP
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

## 🎯 Como Executar

### Menu Interativo (Recomendado)

```bash
cd XAI
python -m xai_fer
```

```
════════════════════════════════════════════════════════════
         XAI ANALYSIS — MENU PRINCIPAL
════════════════════════════════════════════════════════════

  [1] Pipeline Completo (ViT + CNN)
  [2] Pipeline Apenas ViT
  [3] Pipeline Apenas CNN
  [4] Executar LIME/SHAP (imagens estratificadas)
  [5] Regenerar Gráficos (usa CSV existente)
  [6] Regenerar CSVs de Análise
  [0] Sair
```

### Via Linha de Comando

```bash
cd XAI

# Pipeline completo com 1000 imagens
python -m xai_fer --n_samples 1000

# Apenas ViT
python -m xai_fer --models vit --n_samples 500

# Com LIME/SHAP
python -m xai_fer --n_samples 1000 --agnostic --n_agnostic 10

# Regenerar apenas gráficos
python -m xai_fer --plots-only

# Regenerar apenas CSVs de análise
python -m xai_fer --analysis-only

# Apenas LIME/SHAP (usa CSV existente)
python -m xai_fer --agnostic-only
```

### Flags

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

## 📁 Estrutura do Projeto

```
IC-Projeto/
├── Training/
│   └── Models/
│       ├── ViT/              # Checkpoint ViT
│       └── CNN/              # Peso ConvNeXt
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
│   ├── xai_fer/              # Pacote Python principal
│   │   ├── __main__.py       # Entry point
│   │   ├── config.py         # ⚙️ Configurações
│   │   ├── models/           # Explicadores (ViT, CNN, LIME/SHAP)
│   │   ├── evaluation/       # Métricas e análise
│   │   ├── visualization/    # Heatmaps e gráficos
│   │   ├── pipeline/         # Runner e seletor
│   │   └── reporting/        # Tabelas LaTeX
│   ├── experiments.ipynb
│   └── requirements.txt
├── ARCHITECTURE.md            # Decisões e pontos de melhoria
└── README.md
```

## 📊 Métricas Calculadas

### Fidelidade
| Métrica | Descrição |
|---------|-----------|
| **AOPC** | Queda de confiança ao remover regiões importantes |
| **Insertion AUC** | Confiança ao adicionar pixels importantes (↑ melhor) |
| **Deletion AUC** | Confiança ao remover pixels importantes (↓ melhor) |

### Localidade
| Métrica | Descrição |
|---------|-----------|
| **Area@50/90** | % de área para capturar 50%/90% da atenção |
| **Concentration AUC** | AUC da curva massa vs área (↑ mais focado) |
| **Entropy** | Dispersão (↓ mais focado) |
| **Gini** | Concentração (↑ mais focado) |

## 📦 Dependências

- PyTorch ≥ 2.0
- Transformers (HuggingFace)
- timm
- grad-cam
- lime, shap

## 👤 Autor

Tobias Rocha