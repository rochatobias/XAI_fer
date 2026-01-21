# ==============================================================================
# Configuração centralizada do projeto XAI
# ==============================================================================
# Modifique N_SAMPLES para controlar quantas imagens processar
# ==============================================================================

import os
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÃO PRINCIPAL - MODIFIQUE AQUI
# ==============================================================================

# Número de imagens a processar (total, distribuído entre classes)
N_SAMPLES = 2 # <- ALTERE AQUI PARA MUDAR O TAMANHO DA AMOSTRA

# ==============================================================================
# CAMINHOS
# ==============================================================================

# Diretório base do projeto
BASE_DIR = Path(__file__).parent.parent  # XAI/

# Diretório do modelo ViT
MODEL_DIR = str(BASE_DIR.parent / "Training" / "Models" / "ViT" / "deit_fold_5" / "best_checkpoint-45153")

# Diretório dos dados para XAI
DATA_DIR = str(BASE_DIR / "data" / "aplicaçãoXAI")

# Diretório de resultados
RESULTS_DIR = str(BASE_DIR / "results")

# Subdiretórios de resultados
HEATMAPS_DIR = os.path.join(RESULTS_DIR, "heatmaps")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")

# ==============================================================================
# PARÂMETROS DO MODELO E TRANSFORMAÇÕES
# ==============================================================================

IMG_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ==============================================================================
# CONFIGURAÇÃO XAI
# ==============================================================================

XAI_METHODS = ["Raw", "Rollout", "Flow"]
ROLLOUT_RESIDUAL_WEIGHT = 0.5
FLOW_RESIDUAL_WEIGHT = 0.5
FLOW_POWER = 2.0

# ==============================================================================
# CONFIGURAÇÃO DE MÉTRICAS
# ==============================================================================

AOPC_STEPS = (0.1, 0.2, 0.3, 0.5)
INSERTION_DELETION_STEPS = tuple([i / 100 for i in range(0, 101, 5)])
PERTURB_MODE = "mean"
AREA_ALPHAS = (0.50, 0.90)

# ==============================================================================
# CLASSES DE EMOÇÕES
# ==============================================================================

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ==============================================================================
# CONFIGURAÇÃO DE VISUALIZAÇÃO
# ==============================================================================

FIGURE_DPI = 150
OVERLAY_ALPHA = 0.35
DEVICE = None


def get_device():
    """Retorna o device a ser usado (cuda ou cpu)."""
    import torch
    if DEVICE is not None:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_results_dirs():
    """Cria os diretórios de resultados se não existirem."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def print_config():
    """Imprime a configuração atual."""
    print("=" * 60)
    print("CONFIGURAÇÃO XAI")
    print("=" * 60)
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"DEVICE: {get_device()}")
    print(f"XAI_METHODS: {XAI_METHODS}")
    print("=" * 60)

# CNN XAI
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

import timm
from timm.data import resolve_model_data_config, create_transform

import cv2
import matplotlib.pyplot as plt

# PATHS (AJUSTE AQUI)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Modelo CNN + checkpoint do fold 5
MODEL_NAME = "convnext_base"  # igual ao treino
NUM_CLASSES = 7               # ajuste se necessário
FOLD = 5

# checkpoint best do fold 5 (no seu Drive/estrutura)
CNN_CKPT_PATH = Path("/content/drive/MyDrive/cnn_fold_5/checkpoints/convnext_fold_5_best.pth")  # <-- ajuste

# pasta para salvar figuras/resultados (opcional)
OUT_DIR = Path("/content/cnn_xai_fold5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("DEVICE:", DEVICE)
print("CKPT exists:", CNN_CKPT_PATH.exists(), CNN_CKPT_PATH)
print("OUT_DIR:", OUT_DIR)

from torchvision import datasets
from pathlib import Path
import pandas as pd

# =========================
# AJUSTE O PATH DO DATASET
# =========================
DATASET_ROOT = Path("/content/test_dataset")
# ou:
# DATASET_ROOT = Path("/content/drive/MyDrive/dataset")

assert DATASET_ROOT.exists(), f"Path não existe: {DATASET_ROOT}"

# Carrega como ImageFolder
img_ds = datasets.ImageFolder(str(DATASET_ROOT))

print("Classes:", img_ds.classes)
print("Total imagens:", len(img_ds))

# Constrói df no MESMO formato do ViT
rows = []
for path, label in img_ds.samples:
    rows.append({
        "file": str(Path(path).resolve()),
        "label": int(label)
    })

df = pd.DataFrame(rows)

print(df.head())
print("DF size:", len(df))