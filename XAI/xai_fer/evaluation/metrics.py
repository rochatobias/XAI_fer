"""
Métricas de avaliação quantitativa para heatmaps XAI.

Divide-se em dois grupos:

**Métricas de Fidelidade** (dependem do modelo):
    - AOPC (Average drop Of Probability after perturbation Curve): remove os
      pixels mais relevantes segundo o heatmap e mede a queda de confiança.
      Quanto maior o AOPC, mais relevantes eram as regiões destacadas.
    - Insertion AUC: insere progressivamente os pixels mais relevantes em uma
      baseline (média), medindo o crescimento da confiança. Maior = melhor.
    - Deletion AUC: remove progressivamente os pixels mais relevantes,
      medindo a queda de confiança. Menor = melhor.

**Métricas de Localidade** (independem do modelo):
    - Area@α: fração de área que concentra α% da massa do heatmap.
      Menor = mais concentrado/focado.
    - Concentration AUC (MPL): AUC da curva massa vs. proporção de área.
      Maior = mais concentrado.
    - Entropy: entropia de Shannon do heatmap normalizado.
      Menor = mais focado/determinístico.
    - Gini: coeficiente de Gini. Maior = mais concentrado.

Ferramentas utilizadas:
    - PyTorch para perturbação de tensores e forward passes.
    - NumPy para cálculos de localidade (não dependem de GPU).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, Tuple

from xai_fer.config import (
    IMG_SIZE,
    MEAN,
    STD,
    AOPC_STEPS,
    INSERTION_DELETION_STEPS,
)

EPS = 1e-12
_base_tfm = None


# ==============================================================================
# Transform de base
# ==============================================================================


def get_base_transform(img_size: Tuple[int, int] = IMG_SIZE) -> transforms.Compose:
    """Retorna transform base: Resize + ToTensor (sem normalização).

    Cacheia o resultado para evitar recriação repetida.
    """
    global _base_tfm
    if _base_tfm is None:
        _base_tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
    return _base_tfm


def normalize_tensor(
    x_01: torch.Tensor,
    mean: Tuple,
    std: Tuple,
) -> torch.Tensor:
    """Normaliza tensor [0,1] com média e desvio padrão ImageNet."""
    mean_t = torch.tensor(mean, device=x_01.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=x_01.device).view(3, 1, 1)
    return (x_01 - mean_t) / (std_t + EPS)


def _model_forward(
    model: torch.nn.Module,
    x_normalized: torch.Tensor,
    model_type: str = "vit",
) -> torch.Tensor:
    """Forward abstrato: despacha para ViT (``pixel_values``) ou CNN (direto)."""
    if model_type == "vit":
        return model(pixel_values=x_normalized).logits
    return model(x_normalized)


# ==============================================================================
# Perturbação
# ==============================================================================


def perturb_topk(
    x01: torch.Tensor,
    heatmap_224: np.ndarray,
    frac: float,
    mode: str = "mean",
) -> torch.Tensor:
    """Perturba os *top-k* pixels mais relevantes segundo o heatmap.

    Args:
        x01: Tensor [C, H, W] em [0, 1].
        heatmap_224: Heatmap [H, W] em [0, 1].
        frac: Fração de pixels a perturbar.
        mode: ``'mean'`` (substitui pela média da imagem) ou ``'zero'``.

    Returns:
        Tensor perturbado.
    """
    C, H, W = x01.shape
    hm = torch.tensor(heatmap_224, device=x01.device, dtype=torch.float32).view(-1)
    k = int(frac * hm.numel())
    if k <= 0:
        return x01.clone()

    topk_idx = torch.topk(hm, k=k, largest=True).indices
    x_pert = x01.clone().view(C, -1)

    if mode == "mean":
        baseline = x01.mean(dim=(1, 2), keepdim=True).view(C, 1)
        x_pert[:, topk_idx] = baseline
    elif mode == "zero":
        x_pert[:, topk_idx] = 0.0
    else:
        raise ValueError(f"mode deve ser 'mean' ou 'zero', recebido: {mode}")

    return x_pert.view(C, H, W)


def perturb_bottomk(
    x01: torch.Tensor,
    heatmap_224: np.ndarray,
    frac: float,
    insert_from: torch.Tensor,
) -> torch.Tensor:
    """Para Insertion: insere top-k pixels da imagem original na baseline."""
    C, H, W = x01.shape
    hm = torch.tensor(heatmap_224, device=x01.device, dtype=torch.float32).view(-1)
    k = int(frac * hm.numel())
    if k <= 0:
        return insert_from.clone()

    topk_idx = torch.topk(hm, k=k, largest=True).indices
    result = insert_from.clone().view(C, -1)
    original_flat = x01.view(C, -1)
    result[:, topk_idx] = original_flat[:, topk_idx]
    return result.view(C, H, W)


# ==============================================================================
# Métricas de Fidelidade
# ==============================================================================


@torch.no_grad()
def compute_aopc(
    model: torch.nn.Module,
    pil_img: Image.Image,
    heatmap_224: np.ndarray,
    device: torch.device,
    mean: Tuple = MEAN,
    std: Tuple = STD,
    steps: Tuple = AOPC_STEPS,
    perturb_mode: str = "mean",
    model_type: str = "vit",
    true_label_idx: Optional[int] = None,
) -> Dict:
    """Calcula AOPC (Average drop Of Probability after perturbation Curve).

    Remove progressivamente frações dos pixels mais relevantes e mede a
    queda de confiança do modelo na classe predita. Usa baseline selecionável
    (``'mean'`` = média da imagem, ``'zero'`` = preto).

    Args:
        model: Modelo carregado.
        pil_img: Imagem PIL original.
        heatmap_224: Heatmap 224×224 em [0, 1].
        device: Device.
        mean: Média de normalização.
        std: Desvio padrão de normalização.
        steps: Frações de perturbação (ex: 0.1, 0.2, 0.3, 0.5).
        perturb_mode: ``'mean'`` ou ``'zero'``.
        model_type: ``'vit'`` ou ``'cnn'``.
        true_label_idx: Índice da classe verdadeira (não usado no cálculo
                        atual, reservado para futuras extensões).

    Returns:
        Dicionário com ``aopc_mean``, ``drops``, ``confs``, ``base_conf``.
    """
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    base_tfm = get_base_transform()
    x01 = base_tfm(pil_img).to(device)
    x_base = normalize_tensor(x01, mean, std).unsqueeze(0)

    logits = _model_forward(model, x_base, model_type)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    target_idx = pred_idx
    base_conf = float(probs[target_idx].item())

    drops = []
    confs = []
    for f in steps:
        x01_p = perturb_topk(x01, heatmap_224, frac=f, mode=perturb_mode)
        x_p = normalize_tensor(x01_p, mean, std).unsqueeze(0)
        p = F.softmax(_model_forward(model, x_p, model_type), dim=-1).squeeze(0)
        conf = float(p[target_idx].item())
        drop = max(0.0, base_conf - conf)
        confs.append(conf)
        drops.append(drop)

    aopc_mean = float(np.mean(drops)) if drops else 0.0

    return {
        "pred_idx": pred_idx,
        "base_conf": base_conf,
        "drops": drops,
        "confs": confs,
        "aopc_mean": aopc_mean,
        "perturb_mode": perturb_mode,
    }


@torch.no_grad()
def compute_insertion_deletion(
    model: torch.nn.Module,
    pil_img: Image.Image,
    heatmap_224: np.ndarray,
    device: torch.device,
    mean: Tuple = MEAN,
    std: Tuple = STD,
    steps: Tuple = INSERTION_DELETION_STEPS,
    model_type: str = "vit",
    true_label_idx: Optional[int] = None,
) -> Dict:
    """Calcula curvas de Insertion e Deletion.

    Insertion: começa com baseline (média) e adiciona progressivamente
    os pixels mais relevantes → confiança deve subir. Maior AUC = melhor.

    Deletion: começa com imagem completa e remove progressivamente os
    pixels mais relevantes → confiança deve cair. Menor AUC = melhor.

    Returns:
        Dicionário com ``insertion_auc``, ``deletion_auc`` e vetores de confiança.
    """
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    base_tfm = get_base_transform()
    x01 = base_tfm(pil_img).to(device)
    baseline = x01.mean(dim=(1, 2), keepdim=True).expand_as(x01)
    x_base = normalize_tensor(x01, mean, std).unsqueeze(0)

    logits = _model_forward(model, x_base, model_type)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    target_idx = pred_idx

    deletion_confs = []
    insertion_confs = []

    for frac in steps:
        # Deletion
        x_del = perturb_topk(x01, heatmap_224, frac=frac, mode="mean")
        x_del_norm = normalize_tensor(x_del, mean, std).unsqueeze(0)
        p_del = F.softmax(_model_forward(model, x_del_norm, model_type), dim=-1).squeeze(0)
        deletion_confs.append(float(p_del[target_idx].item()))

        # Insertion
        x_ins = perturb_bottomk(x01, heatmap_224, frac=frac, insert_from=baseline)
        x_ins_norm = normalize_tensor(x_ins, mean, std).unsqueeze(0)
        p_ins = F.softmax(_model_forward(model, x_ins_norm, model_type), dim=-1).squeeze(0)
        insertion_confs.append(float(p_ins[target_idx].item()))

    steps_arr = np.array(steps)
    deletion_auc = float(np.trapezoid(deletion_confs, steps_arr))
    insertion_auc = float(np.trapezoid(insertion_confs, steps_arr))

    return {
        "pred_idx": pred_idx,
        "deletion_confs": deletion_confs,
        "insertion_confs": insertion_confs,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
    }


# ==============================================================================
# Métricas de Localidade (model-free)
# ==============================================================================


def ensure_2d(hm: np.ndarray) -> np.ndarray:
    """Garante que heatmap é 2D, colapsando canais se necessário."""
    hm = np.asarray(hm)
    if hm.ndim == 3:
        if hm.shape[0] == 1:
            hm = hm[0]
        elif hm.shape[-1] == 1:
            hm = hm[..., 0]
        else:
            hm = hm.mean(axis=-1)
    return hm.astype(np.float32)


def mass_norm(hm: np.ndarray) -> np.ndarray:
    """Normaliza heatmap para distribuição de massa (soma = 1)."""
    hm = np.maximum(ensure_2d(hm), 0.0)
    s = float(hm.sum())
    if s < EPS:
        return np.zeros_like(hm, dtype=np.float32)
    return (hm / (s + EPS)).astype(np.float32)


def area_at_alpha(hm: np.ndarray, alpha: float) -> float:
    """Fração de área que concentra *α*% da massa.

    Ordena pixels por relevância decrescente e conta quantos são necessários
    para acumular ``alpha`` da massa total. Menor = mais concentrado.
    """
    mass = mass_norm(hm)
    flat = mass.reshape(-1)
    flat_sorted = np.sort(flat)[::-1]
    cumsum = np.cumsum(flat_sorted)
    k = int(np.searchsorted(cumsum, alpha) + 1)
    return float(k / flat.size)


def mpl_curve(
    hm: np.ndarray,
    area_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula curva MPL (Mass vs Proportion of area used to capture it)."""
    mass = mass_norm(hm)
    flat = mass.reshape(-1)
    flat_sorted = np.sort(flat)[::-1]
    cumsum = np.cumsum(flat_sorted)
    n = flat.size
    if area_grid is None:
        area_grid = np.linspace(0.01, 1.0, 100)
    area_grid = np.asarray(area_grid, dtype=np.float32)
    ks = np.clip((area_grid * n).round().astype(int), 1, n)
    mass_captured = np.array([cumsum[k - 1] for k in ks], dtype=np.float32)
    return area_grid, mass_captured


def mpl_auc(hm: np.ndarray) -> float:
    """Calcula AUC da curva MPL (Concentration AUC). Maior = mais concentrado."""
    x, y = mpl_curve(hm)
    return float(np.trapezoid(y, x))


def entropy_map(hm: np.ndarray) -> float:
    """Entropia de Shannon do heatmap. Maior = mais disperso."""
    mass = mass_norm(hm).reshape(-1)
    mass = mass[mass > 0]
    if mass.size == 0:
        return 0.0
    return float(-(mass * np.log(mass + EPS)).sum())


def gini_map(hm: np.ndarray) -> float:
    """Coeficiente de Gini do heatmap. Maior = mais concentrado."""
    x = mass_norm(hm).reshape(-1)
    if x.sum() < EPS:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float32)
    g = (np.sum((2 * idx - n - 1) * x_sorted)) / (n * np.sum(x_sorted) + EPS)
    return float(g)


def compute_locality_metrics(hm: np.ndarray) -> Dict[str, float]:
    """Calcula todas as métricas de localidade de uma só vez."""
    return {
        "Area@50": area_at_alpha(hm, 0.50),
        "Area@90": area_at_alpha(hm, 0.90),
        "Concentration_AUC": mpl_auc(hm),
        "Entropy": entropy_map(hm),
        "Gini": gini_map(hm),
    }


# ==============================================================================
# Função Principal
# ==============================================================================


def compute_all_metrics(
    model: torch.nn.Module,
    pil_img: Image.Image,
    heatmap_224: np.ndarray,
    device: torch.device,
    mean: Tuple = MEAN,
    std: Tuple = STD,
    model_type: str = "vit",
    true_label_idx: Optional[int] = None,
) -> Dict:
    """Calcula todas as métricas (fidelidade + localidade) para um heatmap.

    Executa AOPC com ambas as baselines (``mean`` e ``zero``),
    Insertion/Deletion e métricas de localidade.

    Returns:
        Dicionário plano com todas as métricas prontas para DataFrame.
    """
    aopc_mean_result = compute_aopc(
        model, pil_img, heatmap_224, device, mean, std,
        perturb_mode="mean", model_type=model_type, true_label_idx=true_label_idx,
    )
    aopc_zero_result = compute_aopc(
        model, pil_img, heatmap_224, device, mean, std,
        perturb_mode="zero", model_type=model_type, true_label_idx=true_label_idx,
    )
    ins_del_result = compute_insertion_deletion(
        model, pil_img, heatmap_224, device, mean, std,
        model_type=model_type, true_label_idx=true_label_idx,
    )
    locality = compute_locality_metrics(heatmap_224)

    return {
        "AOPC_mean": aopc_mean_result["aopc_mean"],
        "AOPC_zero": aopc_zero_result["aopc_mean"],
        "Insertion_AUC": ins_del_result["insertion_auc"],
        "Deletion_AUC": ins_del_result["deletion_auc"],
        "base_conf": aopc_mean_result["base_conf"],
        "pred_idx": aopc_mean_result["pred_idx"],
        **locality,
    }
