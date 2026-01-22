"""Configuração centralizada do projeto XAI.

Modifique N_SAMPLES para controlar quantas imagens processar.
"""

import os
from pathlib import Path
import torch

# ==============================================================================
# CONFIGURAÇÃO PRINCIPAL - MODIFIQUE AQUI
# ==============================================================================

# Número de imagens a processar (total, distribuído entre classes)
N_SAMPLES = 1  # <- ALTERE AQUI PARA MUDAR O TAMANHO DA AMOSTRA

# Número de imagens para métodos agnósticos (LIME/SHAP) - são mais lentos
N_SAMPLES_AGNOSTIC = 1  # <- Para LIME/SHAP, pode ser diferente

# ==============================================================================
# CAMINHOS
# ==============================================================================

# Diretório base do projeto
BASE_DIR = Path(__file__).parent.parent  # XAI/
PROJECT_ROOT = BASE_DIR.parent  # IC-Projeto/

# Diretório do modelo ViT (best fold 5)
VIT_MODEL_DIR = str(PROJECT_ROOT / "Training" / "Models" / "ViT" / "best_checkpoint-45153")

# Diretório do modelo CNN (best fold 5)
CNN_MODEL_PATH = str(PROJECT_ROOT / "Training" / "Models" / "CNN" / "convnext_fold_5_best.pth")

# Alias para retrocompatibilidade
MODEL_DIR = VIT_MODEL_DIR

# Diretório dos dados para XAI
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")

# Diretório de resultados
RESULTS_DIR = str(BASE_DIR / "results")

# Subdiretórios de resultados
HEATMAPS_DIR = os.path.join(RESULTS_DIR, "heatmaps")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# ==============================================================================
# PARÂMETROS DO MODELO E TRANSFORMAÇÕES
# ==============================================================================

IMG_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ==============================================================================
# CONFIGURAÇÃO XAI - ViT
# ==============================================================================

VIT_XAI_METHODS = ["Raw", "Rollout", "Flow"]

# Aliases para retrocompatibilidade
XAI_METHODS = VIT_XAI_METHODS

ROLLOUT_RESIDUAL_WEIGHT = 0.5
FLOW_RESIDUAL_WEIGHT = 0.5
FLOW_POWER = 2.0

# ==============================================================================
# CONFIGURAÇÃO XAI - CNN
# ==============================================================================
# LayerCAM é um bom equilíbrio entre velocidade e qualidade
# ScoreCAM é mais robusto mas ~10x mais lento (inviável para 11k imagens)

CNN_XAI_METHODS = ["GradCAM", "GradCAM++", "LayerCAM"]
CNN_MODEL_NAME = "convnext_base"
CNN_NUM_CLASSES = 7

# ==============================================================================
# CONFIGURAÇÃO XAI - Métodos Agnósticos (LIME/SHAP)
# ==============================================================================

AGNOSTIC_XAI_METHODS = ["LIME", "SHAP"]

# ==============================================================================
# CONFIGURAÇÃO DE MÉTRICAS
# ==============================================================================

AOPC_STEPS = (0.1, 0.2, 0.3, 0.5)
INSERTION_DELETION_STEPS = tuple([i / 100 for i in range(0, 101, 5)])
PERTURB_MODES = ["mean", "zero"]  # Ambos os baselines para AOPC
AREA_ALPHAS = (0.50, 0.90)

# ==============================================================================
# CONFIGURAÇÃO DE ANÁLISE DE PESQUISA
# ==============================================================================

# Thresholds para filtros de confiança
HIGH_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD = 0.5

# ==============================================================================
# CLASSES DE EMOÇÕES
# ==============================================================================

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ==============================================================================
# CONFIGURAÇÃO DE VISUALIZAÇÃO E SELEÇÃO DE HEATMAPS
# ==============================================================================

FIGURE_DPI = 150
OVERLAY_ALPHA = 0.35
DEVICE = None

# Estratégia de seleção de heatmaps para evitar sobrecarga de disco
# 'all': Salva para todas as imagens (comportamento original)
# 'stratified': Salva apenas amostras representativas (7 classes x 4 buckets)
# 'none': Não salva heatmaps
HEATMAP_SELECTION_STRATEGY = "stratified"  

# Configuração da estratégia estratificada
HEATMAPS_PER_CELL = 5     # Imagens por combinação (Classe x Bucket)
PERCENTILE_HIGH = 80      # Percentil para alta confiança (P80)
PERCENTILE_LOW = 20       # Percentil para baixa confiança (P20)


def get_device():
    """Retorna o device a ser usado (cuda ou cpu)."""
    if DEVICE is not None:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_results_dirs():
    """Cria os diretórios de resultados se não existirem."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    # Subpastas por modelo
    os.makedirs(os.path.join(HEATMAPS_DIR, "vit"), exist_ok=True)
    os.makedirs(os.path.join(HEATMAPS_DIR, "cnn"), exist_ok=True)


def print_config():
    """Imprime a configuração atual."""
    print("=" * 60)
    print("CONFIGURAÇÃO XAI")
    print("=" * 60)
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"N_SAMPLES_AGNOSTIC: {N_SAMPLES_AGNOSTIC}")
    print(f"VIT_MODEL_DIR: {VIT_MODEL_DIR}")
    print(f"CNN_MODEL_PATH: {CNN_MODEL_PATH}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"DEVICE: {get_device()}")
    print(f"VIT_XAI_METHODS: {VIT_XAI_METHODS}")
    print(f"CNN_XAI_METHODS: {CNN_XAI_METHODS}")
    print(f"AGNOSTIC_XAI_METHODS: {AGNOSTIC_XAI_METHODS}")
    print("=" * 60)