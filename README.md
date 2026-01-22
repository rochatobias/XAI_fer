# XAI para Reconhecimento de Emoções Faciais

Projeto de Explicabilidade (XAI) aplicada a modelos de reconhecimento de emoções faciais, comparando Vision Transformers (ViT) e ConvNeXt (CNN).

## 📋 Descrição

Este projeto implementa e compara métodos de explicabilidade para dois tipos de arquitetura:

### Modelos
- **ViT (Vision Transformer)**: Usando attention maps (Raw, Rollout, Flow)
- **CNN (ConvNeXt)**: Usando métodos CAM (GradCAM, GradCAM++, LayerCAM)

### Métodos XAI Agnósticos (Opcionais)
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations

## 🚀 Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd IC-Projeto

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r XAI/requirements.txt
```

## ⚙️ Configuração

Edite `XAI/scripts/config.py` para ajustar:

```python
N_SAMPLES = 100              # Número de imagens a processar
N_SAMPLES_AGNOSTIC = 10      # Imagens para LIME/SHAP (são mais lentos)
```

### Caminhos dos Modelos
Os modelos treinados devem estar em:
- ViT: `Training/Models/ViT/best_checkpoint-45153/`
- CNN: `Training/Models/CNN/convnext_fold_5_best.pth`

### Dados
Coloque as imagens em `XAI/data/aplicaçãoXAI/` organizadas por classe:
```
aplicaçãoXAI/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

## 🎯 Como Executar

### Pipeline Completo
```bash
cd XAI/scripts
python main.py --n_samples 100 --models vit cnn
```

### Com LIME/SHAP (nas imagens selecionadas)
```bash
python main.py --n_samples 100 --agnostic
```

### Apenas ViT
```bash
python main.py --models vit
```

### Apenas CNN
```bash
python main.py --models cnn
```

### Modo Silencioso
```bash
python main.py --quiet
```

## 📓 Notebook para Experimentos

Use `XAI/experiments.ipynb` para testes individuais:
- Carregar modelos ViT/CNN
- Testar XAI em imagem única
- Visualizar heatmaps inline
- Calcular métricas

## 📁 Estrutura do Projeto

```
IC-Projeto/
├── Training/
│   └── Models/
│       ├── ViT/          # Checkpoint do ViT
│       └── CNN/          # Peso do ConvNeXt
├── XAI/
│   ├── data/
│   │   └── aplicaçãoXAI/ # Imagens para XAI
│   ├── results/
│   │   ├── heatmaps/     # Visualizações geradas
│   │   │   ├── vit/
│   │   │   ├── cnn/
│   │   │   └── agnostic/
│   │   ├── summary/      # Gráficos de resumo
│   │   └── analysis/     # CSVs de análise
│   ├── scripts/
│   │   ├── main.py       # Pipeline principal
│   │   ├── config.py     # Configurações
│   │   ├── vit.py        # XAI para ViT
│   │   ├── cnn.py        # XAI para CNN
│   │   ├── agnostic.py   # LIME e SHAP
│   │   ├── metrics.py    # Métricas de avaliação
│   │   └── ...
│   ├── experiments.ipynb # Notebook para testes
│   └── requirements.txt
└── README.md
```

## 📊 Métricas Calculadas

### Fidelidade
- **AOPC**: Average drop of Probability after perturbation
- **Insertion AUC**: Área sob curva de inserção
- **Deletion AUC**: Área sob curva de deleção

### Localidade
- **Area@50/90**: Fração de área para capturar 50%/90% da massa
- **Gini**: Coeficiente de concentração
- **Entropy**: Dispersão do heatmap

## 🔧 Estratégia de Seleção

O projeto usa seleção estratificada para heatmaps:
- 7 classes × 4 buckets (alta/baixa confiança × acerto/erro)
- Economiza espaço em disco
- Garante representatividade

## 📝 Resultados Gerados

| Arquivo | Descrição |
|---------|-----------|
| `metrics_combined.csv` | Métricas de todos os modelos |
| `heatmap_selection.csv` | Lista de imagens selecionadas |
| `heatmaps/vit/*.png` | Heatmaps do ViT |
| `heatmaps/cnn/*.png` | Heatmaps da CNN |
| `heatmaps/agnostic/*.png` | Heatmaps LIME/SHAP |
| `summary/*.png` | Gráficos de resumo |
| `analysis/*.csv` | Análises por método/classe |

## 📦 Dependências Principais

- PyTorch ≥ 2.0
- Transformers (HuggingFace)
- timm
- grad-cam
- lime, shap (opcional)

## 👤 Autor

Tobias Rocha
