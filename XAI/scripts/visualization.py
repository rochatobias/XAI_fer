# ==============================================================================
# Visualization - Funções de Visualização XAI
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Optional, List
import pandas as pd

from config import IMG_SIZE, OVERLAY_ALPHA, FIGURE_DPI


def save_xai_visualization(
    pil_img: Image.Image,
    maps_dict: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
    show: bool = False,
    overlay_alpha: float = OVERLAY_ALPHA,
    dpi: int = FIGURE_DPI
) -> None:
    """
    Salva visualização com imagem original e heatmaps XAI.
    
    Args:
        pil_img: Imagem PIL original
        maps_dict: Dicionário {método: heatmap_224x224}
        save_path: Caminho para salvar a figura
        title: Título da figura
        show: Se True, mostra a figura
        overlay_alpha: Transparência do overlay
        dpi: DPI da figura salva
    """
    img = pil_img.resize(IMG_SIZE)
    img_np = np.array(img)
    
    n = len(maps_dict)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    
    if n == 0:
        axes = [axes]
    
    # Imagem original
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original")
    
    # Heatmaps
    for i, (name, heatmap) in enumerate(maps_dict.items(), 1):
        axes[i].imshow(img_np)
        axes[i].imshow(heatmap, alpha=overlay_alpha, cmap='jet')
        axes[i].axis("off")
        axes[i].set_title(name)
    
    if title:
        fig.suptitle(title, fontsize=10)
    
    plt.tight_layout()
    
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_summary_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    show: bool = False
) -> None:
    """
    Gera gráficos de sumário dos resultados.
    
    Args:
        results_df: DataFrame com resultados
        output_dir: Diretório para salvar figuras
        show: Se True, mostra as figuras
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Métricas por método XAI
    _plot_metrics_by_method(results_df, output_dir, show)
    
    # 2. Distribuição de confiança
    _plot_confidence_distribution(results_df, output_dir, show)
    
    # 3. Acurácia por classe
    _plot_accuracy_by_class(results_df, output_dir, show)


def _plot_metrics_by_method(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Plota métricas médias por método XAI."""
    metric_cols = [c for c in df.columns if c in [
        "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC"
    ]]
    
    # Fallback para nomes antigos
    if not metric_cols:
        metric_cols = [c for c in df.columns if c in [
            "AOPC", "Insertion_AUC", "Deletion_AUC"
        ]]
    
    if not metric_cols or "method" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df["method"].unique()
    x = np.arange(len(methods))
    width = 0.8 / len(metric_cols)
    
    for i, col in enumerate(metric_cols):
        means = [df[df["method"] == m][col].mean() for m in methods]
        stds = [df[df["method"] == m][col].std() for m in methods]
        ax.bar(x + i * width, means, width, label=col, yerr=stds, capsize=3)
    
    ax.set_xlabel("Método XAI")
    ax.set_ylabel("Valor da Métrica")
    ax.set_title("Métricas por Método XAI")
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_method.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_confidence_distribution(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Plota distribuição de confiança por acerto/erro."""
    if "conf" not in df.columns or "correct" not in df.columns:
        return
    
    # Agrupa por imagem (evita duplicatas por método)
    df_unique = df.groupby("image_idx").agg({
        "conf": "first",
        "correct": "first"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    correct_conf = df_unique[df_unique["correct"]]["conf"]
    error_conf = df_unique[~df_unique["correct"]]["conf"]
    
    ax.hist(correct_conf, bins=20, alpha=0.7, label=f"Correto (n={len(correct_conf)})", color="green")
    ax.hist(error_conf, bins=20, alpha=0.7, label=f"Erro (n={len(error_conf)})", color="red")
    
    ax.set_xlabel("Confiança")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de Confiança por Resultado")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_accuracy_by_class(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Plota acurácia por classe de emoção."""
    if "label" not in df.columns or "correct" not in df.columns:
        return
    
    # Agrupa por imagem
    df_unique = df.groupby("image_idx").agg({
        "label": "first",
        "correct": "first"
    }).reset_index()
    
    accuracy_by_class = df_unique.groupby("label")["correct"].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = accuracy_by_class.index.tolist()
    accuracies = accuracy_by_class.values
    
    ax.bar(classes, accuracies, color="steelblue")
    ax.axhline(y=accuracy_by_class.mean(), color="red", linestyle="--", label=f"Média: {accuracy_by_class.mean():.2%}")
    
    ax.set_xlabel("Classe")
    ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia por Classe de Emoção")
    ax.set_ylim(0, 1)
    ax.legend()
    
    for i, (c, acc) in enumerate(zip(classes, accuracies)):
        ax.text(i, acc + 0.02, f"{acc:.0%}", ha="center", fontsize=9)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_class.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()
