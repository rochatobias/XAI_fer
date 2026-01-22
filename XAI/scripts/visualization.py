# ==============================================================================
# Visualization - Funções de Visualização XAI
# ==============================================================================
# Heatmaps com colormap turbo (azul→amarelo) e threshold para fundo limpo
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from typing import Dict, Optional, List, Tuple
import pandas as pd

from config import IMG_SIZE, OVERLAY_ALPHA, FIGURE_DPI

# Threshold para mascarar valores muito baixos do heatmap (reduz o azul do fundo)
HEATMAP_THRESHOLD = 0.10  # 10% - valores abaixo disso ficam transparentes


def create_masked_heatmap(heatmap: np.ndarray, threshold: float = HEATMAP_THRESHOLD) -> np.ma.MaskedArray:
    """
    Cria heatmap mascarado onde valores < threshold são transparentes.
    Isso reduz o fundo azulado mantendo as áreas de atenção visíveis.
    """
    # Normaliza para [0, 1]
    hm = heatmap.copy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    # Mascara valores baixos (abaixo do threshold)
    masked = np.ma.masked_where(hm < threshold, hm)
    return masked


def get_turbo_colormap():
    """
    Retorna colormap 'turbo' (azul→verde→amarelo→vermelho).
    Similar ao jet mas com melhor percepção visual.
    Valores masked ficam transparentes.
    """
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(alpha=0)  # Valores masked são transparentes
    return cmap


def save_xai_visualization(
    pil_img: Image.Image,
    maps_dict: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
    show: bool = False,
    overlay_alpha: float = OVERLAY_ALPHA,  # 0.35 por padrão
    dpi: int = FIGURE_DPI,
    use_threshold: bool = True
) -> None:
    """
    Salva visualização com imagem original e heatmaps XAI.
    
    Args:
        use_threshold: Se True, mascara valores baixos (menos azul no fundo)
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
    
    # Heatmaps com colormap turbo
    cmap = get_turbo_colormap() if use_threshold else plt.cm.turbo
    
    for i, (name, heatmap) in enumerate(maps_dict.items(), 1):
        axes[i].imshow(img_np)
        
        if use_threshold:
            masked_hm = create_masked_heatmap(heatmap)
            axes[i].imshow(masked_hm, alpha=overlay_alpha, cmap=cmap, vmin=0, vmax=1)
        else:
            # Sem threshold - normaliza e aplica diretamente
            hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            axes[i].imshow(hm_norm, alpha=overlay_alpha, cmap='turbo')
        
        axes[i].axis("off")
        axes[i].set_title(name)
    
    if title:
        fig.suptitle(title, fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    
    if show:
        plt.show()
    else:
        plt.close()


def save_comparison_visualization(
    pil_img: Image.Image,
    vit_maps: Dict[str, np.ndarray],
    cnn_maps: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
    show: bool = False
) -> None:
    """
    Salva comparação lado-a-lado ViT vs CNN.
    """
    img = pil_img.resize(IMG_SIZE)
    img_np = np.array(img)
    
    # Seleciona melhor método de cada modelo (por concentração)
    def get_best_map(maps: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        best_name, best_map = None, None
        best_concentration = -1
        for name, hm in maps.items():
            hm_norm = hm / (hm.sum() + 1e-8)
            concentration = np.max(hm_norm)
            if concentration > best_concentration:
                best_concentration = concentration
                best_name, best_map = name, hm
        return best_name, best_map
    
    vit_best_name, vit_best = get_best_map(vit_maps) if vit_maps else ("N/A", np.zeros(IMG_SIZE))
    cnn_best_name, cnn_best = get_best_map(cnn_maps) if cnn_maps else ("N/A", np.zeros(IMG_SIZE))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = get_turbo_colormap()
    
    # Original
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original")
    
    # ViT Best
    axes[1].imshow(img_np)
    if vit_maps:
        masked_vit = create_masked_heatmap(vit_best)
        axes[1].imshow(masked_vit, alpha=OVERLAY_ALPHA, cmap=cmap, vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].set_title(f"ViT ({vit_best_name})")
    
    # CNN Best
    axes[2].imshow(img_np)
    if cnn_maps:
        masked_cnn = create_masked_heatmap(cnn_best)
        axes[2].imshow(masked_cnn, alpha=OVERLAY_ALPHA, cmap=cmap, vmin=0, vmax=1)
    axes[2].axis("off")
    axes[2].set_title(f"CNN ({cnn_best_name})")
    
    if title:
        fig.suptitle(title, fontsize=11)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_summary_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    show: bool = False
) -> None:
    """Gera gráficos de sumário dos resultados."""
    os.makedirs(output_dir, exist_ok=True)
    
    _plot_metrics_by_method(results_df, output_dir, show)
    _plot_confidence_distribution(results_df, output_dir, show)
    _plot_accuracy_by_class(results_df, output_dir, show)
    _plot_model_comparison(results_df, output_dir, show)
    _plot_metrics_radar(results_df, output_dir, show)


def _plot_metrics_by_method(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Plota métricas médias por método XAI."""
    metric_cols = [c for c in df.columns if c in [
        "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC"
    ]]
    
    if not metric_cols or "method" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df["method"].unique()
    x = np.arange(len(methods))
    width = 0.8 / len(metric_cols)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, col in enumerate(metric_cols):
        means = [df[df["method"] == m][col].mean() for m in methods]
        stds = [df[df["method"] == m][col].std() for m in methods]
        ax.bar(x + i * width, means, width, label=col, yerr=stds, capsize=3, color=colors[i % len(colors)])
    
    ax.set_xlabel("Método XAI", fontsize=12)
    ax.set_ylabel("Valor da Métrica", fontsize=12)
    ax.set_title("Métricas de Fidelidade por Método XAI", fontsize=14)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_method.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_model_comparison(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Compara métricas entre modelos (ViT vs CNN)."""
    if "model" not in df.columns:
        return
    
    metric_cols = ["AOPC_mean", "Insertion_AUC", "Deletion_AUC", "Gini"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    if not metric_cols:
        return
    
    models = df["model"].unique()
    
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(metric_cols):
        values = [df[df["model"] == m][col].mean() for m in models]
        stds = [df[df["model"] == m][col].std() for m in models]
        
        colors = ['#3498db', '#e74c3c'][:len(models)]
        axes[i].bar(models, values, yerr=stds, capsize=5, color=colors)
        axes[i].set_title(col, fontsize=12)
        axes[i].grid(axis='y', alpha=0.3)
    
    fig.suptitle("Comparação ViT vs CNN", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_metrics_radar(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Gráfico de radar comparando métodos XAI."""
    if "method" not in df.columns:
        return
    
    metrics = ["AOPC_mean", "Insertion_AUC", "Gini", "Area@50"]
    metrics = [m for m in metrics if m in df.columns]
    
    if len(metrics) < 3:
        return
    
    methods = df["method"].unique()[:6]
    
    values_dict = {}
    for method in methods:
        method_df = df[df["method"] == method]
        values = []
        for m in metrics:
            val = method_df[m].mean()
            min_val = df[m].min()
            max_val = df[m].max()
            norm_val = (val - min_val) / (max_val - min_val + 1e-8)
            values.append(norm_val)
        values_dict[method] = values
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, (method, values) in enumerate(values_dict.items()):
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values_closed, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title("Comparação de Métodos XAI (Normalizado)", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_radar.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_confidence_distribution(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """Plota distribuição de confiança por acerto/erro."""
    if "conf" not in df.columns or "correct" not in df.columns:
        return
    
    df_unique = df.groupby("image_idx").agg({
        "conf": "first",
        "correct": "first"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    correct_conf = df_unique[df_unique["correct"]]["conf"]
    error_conf = df_unique[~df_unique["correct"]]["conf"]
    
    ax.hist(correct_conf, bins=20, alpha=0.7, label=f"Correto (n={len(correct_conf)})", color="#2ecc71")
    ax.hist(error_conf, bins=20, alpha=0.7, label=f"Erro (n={len(error_conf)})", color="#e74c3c")
    
    ax.set_xlabel("Confiança", fontsize=12)
    ax.set_ylabel("Frequência", fontsize=12)
    ax.set_title("Distribuição de Confiança por Resultado", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
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
    
    df_unique = df.groupby("image_idx").agg({
        "label": "first",
        "correct": "first"
    }).reset_index()
    
    accuracy_by_class = df_unique.groupby("label")["correct"].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = accuracy_by_class.index.tolist()
    accuracies = accuracy_by_class.values
    
    colors = ['#3498db' if acc >= 0.7 else '#e74c3c' for acc in accuracies]
    ax.bar(classes, accuracies, color=colors)
    ax.axhline(y=accuracy_by_class.mean(), color="gray", linestyle="--", 
               label=f"Média: {accuracy_by_class.mean():.1%}")
    
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Acurácia", fontsize=12)
    ax.set_title("Acurácia por Classe de Emoção", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (c, acc) in enumerate(zip(classes, accuracies)):
        ax.text(i, acc + 0.03, f"{acc:.0%}", ha="center", fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_class.png"), dpi=FIGURE_DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_insertion_deletion_curves(df: pd.DataFrame, output_dir: str, show: bool = False) -> None:
    """
    Plota curvas médias de Insertion e Deletion para cada modelo e método.
    Isso é fundamental para o paper (mostra a qualidade do heatmap).
    """
    # Verifica se temos os dados das curvas (estão nas colunas como strings/listas)
    if not all(c in df.columns for c in ["insertion_confs", "deletion_confs"]):
        return

    # Helper para converter string de lista em np array
    def parse_array(x):
        if isinstance(x, str):
            return np.array(eval(x))
        return np.array(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Agrupa por (Modelo, Método)
    groups = df.groupby(["model", "method"])
    
    for i, ((model_name, method_name), group) in enumerate(groups):
        # Média das curvas
        ins_curves = np.stack([parse_array(x) for x in group["insertion_confs"].values])
        del_curves = np.stack([parse_array(x) for x in group["deletion_confs"].values])
        
        ins_mean = ins_curves.mean(axis=0)
        del_mean = del_curves.mean(axis=0)
        
        steps = np.linspace(0, 1, len(ins_mean))
        
        label = f"{model_name}-{method_name}"
        color = colors[i % len(colors)]
        
        # Insertion (Higher is better)
        axes[0].plot(steps, ins_mean, label=label, color=color, linewidth=2)
        
        # Deletion (Lower is better)
        axes[1].plot(steps, del_mean, label=label, color=color, linewidth=2, linestyle='--')

    # Configuração Insertion
    axes[0].set_title("Insertion Curve (Higher is Better)", fontsize=14)
    axes[0].set_xlabel("Fração de Pixels Inseridos", fontsize=12)
    axes[0].set_ylabel("Probabilidade da Classe", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Configuração Deletion
    axes[1].set_title("Deletion Curve (Lower is Better)", fontsize=14)
    axes[1].set_xlabel("Fração de Pixels Removidos", fontsize=12)
    axes[1].set_ylabel("Probabilidade da Classe", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "insertion_deletion_curves.png"), dpi=FIGURE_DPI)
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_summary_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    show: bool = False
) -> None:
    """Gera gráficos de sumário dos resultados."""
    os.makedirs(output_dir, exist_ok=True)
    
    _plot_metrics_by_method(results_df, output_dir, show)
    _plot_confidence_distribution(results_df, output_dir, show)
    _plot_accuracy_by_class(results_df, output_dir, show)
    _plot_model_comparison(results_df, output_dir, show)
    _plot_metrics_radar(results_df, output_dir, show)
    
    # Novos plots
    try:
        _plot_insertion_deletion_curves(results_df, output_dir, show)
    except Exception as e:
        print(f"Erro ao plotar curvas Insertion/Deletion: {e}")

