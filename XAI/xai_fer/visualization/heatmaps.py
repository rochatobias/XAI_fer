"""
Visualização de heatmaps individuais (overlay sobre imagens).

Funções para salvar visualizações de heatmaps XAI como imagens PNG,
com colormaps específicos por tipo de método:

- **ViT/CNN**: colormap ``turbo`` (azul→verde→amarelo→vermelho) com
  threshold para tornar o fundo (valores < 10%) transparente, evitando
  a dominância de azul.
- **LIME**: colormap ``Greens`` (tons de verde), pois o heatmap contém
  apenas contribuições positivas.
- **SHAP**: colormap ``RdBu_r`` (vermelho = positivo, azul = negativo),
  pois o heatmap está em [-1, 1] com escala simétrica.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple

from xai_fer.config import IMG_SIZE, OVERLAY_ALPHA, FIGURE_DPI

# Valores abaixo deste threshold ficam transparentes no overlay
HEATMAP_THRESHOLD = 0.10


def create_masked_heatmap(
    heatmap: np.ndarray,
    threshold: float = HEATMAP_THRESHOLD,
) -> np.ma.MaskedArray:
    """Cria heatmap mascarado onde valores < threshold são transparentes.

    Isso reduz o fundo azulado (do colormap turbo) mantendo apenas
    as regiões de atenção realmente relevantes.
    """
    hm = heatmap.copy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return np.ma.masked_where(hm < threshold, hm)


def get_turbo_colormap():
    """Retorna colormap 'turbo' com valores masked transparentes.

    Turbo é similar ao jet mas com melhor percepção visual e transições
    mais suaves. Valores masked (abaixo do threshold) ficam com alpha=0.
    """
    cmap = plt.cm.turbo.copy()
    cmap.set_bad(alpha=0)
    return cmap


def save_xai_visualization(
    pil_img: Image.Image,
    maps_dict: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
    show: bool = False,
    overlay_alpha: float = OVERLAY_ALPHA,
    dpi: int = FIGURE_DPI,
    use_threshold: bool = True,
) -> None:
    """Salva visualização com imagem original + overlays de heatmaps XAI.

    Gera uma figura com N+1 subplots: a imagem original à esquerda e
    cada heatmap como overlay à direita. O colormap é selecionado
    automaticamente conforme o método (turbo/Greens/RdBu_r).

    **Gráfico gerado**: grid 1×(N+1) com a imagem original e cada mapa
    XAI sobreposto com transparência. Usado para inspeção visual rápida
    de quais regiões o modelo está olhando em cada método.

    Args:
        pil_img: Imagem PIL original.
        maps_dict: Dicionário ``{nome_método: heatmap_224x224}``.
        save_path: Caminho de saída da imagem PNG.
        title: Título da figura (ex: ``'VIT | OK | True=happy | ...'``).
        show: Se ``True``, exibe a figura interativamente.
        overlay_alpha: Transparência do overlay (0 = invisível, 1 = opaco).
        dpi: Resolução de saída.
        use_threshold: Se ``True``, mascara valores baixos do heatmap.
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

    # Heatmaps com colormaps específicos
    for i, (name, heatmap) in enumerate(maps_dict.items(), 1):
        axes[i].imshow(img_np)
        name_upper = name.upper()

        if name_upper == "LIME":
            axes[i].imshow(heatmap, alpha=0.65, cmap="Greens", vmin=0, vmax=1)
        elif name_upper == "SHAP":
            axes[i].imshow(heatmap, alpha=0.6, cmap="RdBu_r", vmin=-1, vmax=1)
        else:
            cmap = get_turbo_colormap() if use_threshold else plt.cm.turbo
            if use_threshold:
                masked_hm = create_masked_heatmap(heatmap)
                axes[i].imshow(masked_hm, alpha=overlay_alpha, cmap=cmap, vmin=0, vmax=1)
            else:
                hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                axes[i].imshow(hm_norm, alpha=overlay_alpha, cmap="turbo")

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
    show: bool = False,
) -> None:
    """Salva comparação lado-a-lado ViT vs CNN (melhor método de cada).

    Seleciona o mapa com maior concentração (max do heatmap normalizado)
    de cada modelo e exibe em 3 subplots: Original | ViT | CNN.

    **Gráfico gerado**: grid 1×3 comparando a imagem original com o
    melhor heatmap de cada arquitetura. Útil para visualizar diferenças
    de foco entre ViT (atenção global) e CNN (features locais).
    """
    img = pil_img.resize(IMG_SIZE)
    img_np = np.array(img)

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

    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(img_np)
    if vit_maps:
        masked_vit = create_masked_heatmap(vit_best)
        axes[1].imshow(masked_vit, alpha=OVERLAY_ALPHA, cmap=cmap, vmin=0, vmax=1)
    axes[1].axis("off")
    axes[1].set_title(f"ViT ({vit_best_name})")

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
