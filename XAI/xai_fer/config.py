"""
Configuração centralizada do projeto XAI-FER.

Centraliza todos os hiperparâmetros, caminhos de modelos/dados e constantes
utilizadas pelo pipeline. Modifique N_SAMPLES e os caminhos de modelo para
adaptar a execução ao seu ambiente.

Decisões:
    - pathlib.Path para todos os caminhos (consistência, portabilidade).
    - Constantes de normalização ImageNet (0.485, …) porque ambos os modelos
      (ViT fine-tuned e ConvNeXt) foram pré-treinados no ImageNet.
    - AOPC_STEPS e INSERTION_DELETION_STEPS definem a granularidade das curvas
      de fidelidade; valores mais finos (5 %) dão curvas suaves para o paper.
"""

import os
from pathlib import Path

import torch


# ==============================================================================
# CONFIGURAÇÃO PRINCIPAL — MODIFIQUE AQUI
# ==============================================================================

N_SAMPLES: int = 1              # Imagens totais para ViT/CNN (distribuídas entre classes)
N_SAMPLES_AGNOSTIC: int = 1     # Imagens para LIME/SHAP (mais lentos)

# Caminhos base
BASE_DIR = Path(__file__).resolve().parent.parent        # XAI/
PROJECT_ROOT = BASE_DIR.parent                           # IC-Projeto/

# ============ MODELOS ============
VIT_MODEL_DIR = str(PROJECT_ROOT / "Training" / "Models" / "ViT" / "best_checkpoint-45153")
CNN_MODEL_PATH = str(PROJECT_ROOT / "Training" / "Models" / "CNN" / "convnext_fold_5_best.pth")

# ============ DATASET ============
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")

# Diretórios de saída (criados automaticamente)
RESULTS_DIR  = str(BASE_DIR / "results")
HEATMAPS_DIR = os.path.join(RESULTS_DIR, "heatmaps")
SUMMARY_DIR  = os.path.join(RESULTS_DIR, "summary")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# ==============================================================================
# PARÂMETROS DE PRÉ-PROCESSAMENTO
# ==============================================================================

IMG_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ==============================================================================
# MÉTODOS XAI
# ==============================================================================

# ViT — baseados em atenção
VIT_XAI_METHODS = ["Raw", "Rollout", "Flow"]
ROLLOUT_RESIDUAL_WEIGHT = 0.5
FLOW_RESIDUAL_WEIGHT    = 0.5
FLOW_POWER              = 2.0

# CNN — CAM-based (LayerCAM equilibra velocidade e qualidade)
CNN_XAI_METHODS = ["GradCAM", "GradCAM++", "LayerCAM"]
CNN_MODEL_NAME  = "convnext_base"
CNN_NUM_CLASSES = 7

# Agnósticos
AGNOSTIC_XAI_METHODS = ["LIME", "SHAP"]

# ==============================================================================
# MÉTRICAS DE FIDELIDADE
# ==============================================================================

AOPC_STEPS               = (0.1, 0.2, 0.3, 0.5)
INSERTION_DELETION_STEPS = tuple(i / 100 for i in range(0, 101, 5))
PERTURB_MODES            = ["mean", "zero"]
AREA_ALPHAS              = (0.50, 0.90)

# ==============================================================================
# THRESHOLDS DE CONFIANÇA
# ==============================================================================

HIGH_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_THRESHOLD  = 0.5

# ==============================================================================
# CLASSES
# ==============================================================================

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

FIGURE_DPI    = 150
OVERLAY_ALPHA = 0.35
DEVICE        = None   # None = autodetect

# ==============================================================================
# SELEÇÃO ESTRATIFICADA
# ==============================================================================

HEATMAP_SELECTION_STRATEGY = "stratified"   # 'all', 'stratified', 'none'
HEATMAPS_PER_CELL          = 5             # Imagens por (Classe × Bucket)
PERCENTILE_HIGH            = 80
PERCENTILE_LOW             = 20


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def get_device() -> torch.device:
    """Retorna o device a ser usado (CUDA se disponível, senão CPU)."""
    if DEVICE is not None:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_results_dirs() -> None:
    """Cria a árvore de diretórios de resultados, incluindo subpastas por modelo."""
    for d in (RESULTS_DIR, HEATMAPS_DIR, SUMMARY_DIR, ANALYSIS_DIR):
        os.makedirs(d, exist_ok=True)
    for sub in ("vit", "cnn"):
        os.makedirs(os.path.join(HEATMAPS_DIR, sub), exist_ok=True)


def print_config() -> None:
    """Imprime a configuração atual no console."""
    print("=" * 60)
    print("CONFIGURAÇÃO XAI")
    print("=" * 60)
    print(f"N_SAMPLES:            {N_SAMPLES}")
    print(f"N_SAMPLES_AGNOSTIC:   {N_SAMPLES_AGNOSTIC}")
    print(f"VIT_MODEL_DIR:        {VIT_MODEL_DIR}")
    print(f"CNN_MODEL_PATH:       {CNN_MODEL_PATH}")
    print(f"DATA_DIR:             {DATA_DIR}")
    print(f"RESULTS_DIR:          {RESULTS_DIR}")
    print(f"DEVICE:               {get_device()}")
    print(f"VIT_XAI_METHODS:      {VIT_XAI_METHODS}")
    print(f"CNN_XAI_METHODS:      {CNN_XAI_METHODS}")
    print(f"AGNOSTIC_XAI_METHODS: {AGNOSTIC_XAI_METHODS}")
    print("=" * 60)
