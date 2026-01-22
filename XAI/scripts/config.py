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

# Caminhos base
BASE_DIR = Path(__file__).parent.parent  # XAI/
PROJECT_ROOT = BASE_DIR.parent  # IC-Projeto/

# ============ MODELOS - MODIFIQUE AQUI SE NECESSÁRIO ============
VIT_MODEL_DIR = str(PROJECT_ROOT / "Training" / "Models" / "ViT" / "best_checkpoint-45153")
CNN_MODEL_PATH = str(PROJECT_ROOT / "Training" / "Models" / "CNN" / "convnext_fold_5_best.pth")

# ============ DATASET - MODIFIQUE AQUI SE NECESSÁRIO ============
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")

# Diretórios de saída (criados automaticamente)
RESULTS_DIR = str(BASE_DIR / "results")
HEATMAPS_DIR = os.path.join(RESULTS_DIR, "heatmaps")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# ==============================================================================
# PARÂMETROS DO MODELO E TRANSFORMAÇÕES
# ==============================================================================

IMG_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Métodos XAI - ViT (baseados em atenção)
VIT_XAI_METHODS = ["Raw", "Rollout", "Flow"]
ROLLOUT_RESIDUAL_WEIGHT = 0.5
FLOW_RESIDUAL_WEIGHT = 0.5
FLOW_POWER = 2.0

# Métodos XAI - CNN (CAM-based)
# LayerCAM: bom equilíbrio velocidade/qualidade | ScoreCAM: mais robusto mas ~10x mais lento
CNN_XAI_METHODS = ["GradCAM", "GradCAM++", "LayerCAM"]
CNN_MODEL_NAME = "convnext_base"
CNN_NUM_CLASSES = 7

# ==============================================================================
# CONFIGURAÇÃO XAI - Métodos Agnósticos (LIME/SHAP)
# ==============================================================================
# Métodos agnósticos (LIME/SHAP) - mais lentos, apenas nas imagens estratificadas
AGNOSTIC_XAI_METHODS = ["LIME", "SHAP"]

# Métricas de fidelidade
AOPC_STEPS = (0.1, 0.2, 0.3, 0.5)
INSERTION_DELETION_STEPS = tuple([i / 100 for i in range(0, 101, 5)])
PERTURB_MODES = ["mean", "zero"]
AREA_ALPHAS = (0.50, 0.90)

# Thresholds para filtros de confiança
HIGH_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD = 0.5

# Classes de emoções (7 classes FER)
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Visualização
FIGURE_DPI = 150
OVERLAY_ALPHA = 0.35
DEVICE = None

# Seleção estratificada de heatmaps: 7 classes x 4 buckets x N imagens
HEATMAP_SELECTION_STRATEGY = "stratified"  # 'all', 'stratified', 'none'
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