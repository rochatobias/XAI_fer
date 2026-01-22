"""ViT XAI - Métodos de Explicabilidade para Vision Transformers."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoConfig
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config
from typing import Tuple, Dict, Optional

from config import (
    VIT_MODEL_DIR, ROLLOUT_RESIDUAL_WEIGHT, FLOW_RESIDUAL_WEIGHT, 
    FLOW_POWER, OVERLAY_ALPHA, get_device
)

EPS = 1e-12

def _avg_heads(attn_layer: torch.Tensor) -> torch.Tensor:
    """Média das atenções sobre todas as heads."""
    return attn_layer[0].mean(dim=0)


def _cls_to_patch_map(attn_TT: torch.Tensor) -> torch.Tensor:
    """Extrai atenção do CLS para patches."""
    return attn_TT[0, 1:]


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def load_vit_model(model_dir: str = VIT_MODEL_DIR, device: Optional[str] = None) -> Tuple:
    """Carrega o modelo ViT para classificação com suporte a atenções."""
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    ).to(device)
    model.config.output_attentions = True
    model.eval()
    print(f"Model type: {cfg.model_type} | num_labels: {getattr(cfg, 'num_labels', None)}")
    print(f"Model: {type(model).__name__} | device: {device}")
    return model, cfg, device


def build_transform_from_convnext() -> Tuple:
    """Constrói transformações de pré-processamento baseadas no ConvNeXt."""
    temp_model = timm.create_model("convnext_base", pretrained=True, num_classes=1000)
    data_cfg = resolve_model_data_config(temp_model)
    mean = tuple(data_cfg["mean"])
    std = tuple(data_cfg["std"])
    size = tuple(data_cfg["input_size"][1:])
    del temp_model
    torch.cuda.empty_cache()
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    print(f"Transform: size={size}, mean={mean}, std={std}")
    return tfm, size, mean, std


def load_image_as_tensor(image_path: str, transform, device) -> Tuple[Image.Image, torch.Tensor]:
    """Carrega uma imagem e aplica as transformações."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return img, x


@torch.no_grad()
def forward_with_attentions(model, x: torch.Tensor) -> Tuple:
    """Forward pass com retorno de atenções."""
    out = model(pixel_values=x, output_attentions=True)
    logits = out.logits
    probs = F.softmax(logits, dim=-1)
    pred = int(probs.argmax(dim=-1).item())
    conf = float(probs.max(dim=-1).values.item())
    attentions = out.attentions
    return pred, conf, probs.squeeze(0).cpu().numpy(), attentions


def xai_raw_attention(attentions: Tuple) -> np.ndarray:
    """Raw Attention: média das heads da última camada."""
    last = attentions[-1]
    A = _avg_heads(last)
    v = _cls_to_patch_map(A)
    return v.detach().cpu().numpy()


def xai_attention_rollout(attentions: Tuple, residual_weight: float = ROLLOUT_RESIDUAL_WEIGHT) -> np.ndarray:
    """Attention Rollout: multiplica atenções camada a camada."""
    L = len(attentions)
    A0 = _avg_heads(attentions[0])  # [T, T]
    T = A0.shape[-1]
    I = torch.eye(T, device=A0.device, dtype=A0.dtype)

    R = I  # acumulador
    for l in range(L):
        A = _avg_heads(attentions[l])              # [T, T]
        A = residual_weight * A + (1.0 - residual_weight) * I
        A = A / (A.sum(dim=-1, keepdim=True) + EPS)

        # composição (ordem recomendada)
        R = R @ A

    v = R[0, 1:]  # CLS -> patches
    v = v / (v.sum() + EPS)  # normaliza para estabilidade/compatibilidade
    return v.detach().cpu().numpy()


def xai_attention_flow(attentions: Tuple, residual_weight: float = FLOW_RESIDUAL_WEIGHT, power: float = FLOW_POWER) -> np.ndarray:
    """Attention Flow: propagação de influência do CLS."""
    L = len(attentions)
    A0 = _avg_heads(attentions[0])  # [T, T]
    T = A0.shape[-1]
    I = torch.eye(T, device=A0.device, dtype=A0.dtype)

    f = torch.zeros(T, device=A0.device, dtype=A0.dtype)
    f[0] = 1.0  # massa no CLS

    for l in range(L):
        A = _avg_heads(attentions[l])              # [T, T]
        A = torch.pow(A, power)                    # destaca conexões fortes (se power>1)
        A = residual_weight * A + (1.0 - residual_weight) * I
        A = A / (A.sum(dim=-1, keepdim=True) + EPS)

        f = f @ A

    v = f[1:]  # patches
    v = v / (v.sum() + EPS)  # normaliza
    return v.detach().cpu().numpy()


def vector_to_patchmap(v_196: np.ndarray, grid_size: int = 14, p: float = 99.0) -> np.ndarray:
    m = v_196.reshape(grid_size, grid_size).astype(np.float32)
    m = m - m.min()
    denom = np.percentile(m, p) + 1e-8
    m = np.clip(m / denom, 0, 1)
    return m


def upscale_to_image(map_14: np.ndarray, out_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Aumenta resolução do mapa para tamanho da imagem."""
    m = (map_14 * 255).astype(np.uint8)
    m_img = Image.fromarray(m).resize(out_size, resample=Image.BILINEAR)
    return np.array(m_img) / 255.0


def plot_xai_overlays(pil_img: Image.Image, maps_dict: Dict[str, np.ndarray], title_prefix: str = "XAI", save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    """Plota imagem original com overlays dos mapas XAI."""
    img = pil_img.resize((224, 224))
    img_np = np.array(img)
    n = len(maps_dict)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title(f"{title_prefix} - Original")
    for i, (name, m224) in enumerate(maps_dict.items(), 1):
        axes[i].imshow(img_np)
        axes[i].imshow(m224, alpha=OVERLAY_ALPHA, cmap='jet')
        axes[i].axis("off")
        axes[i].set_title(f"{title_prefix} - {name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def run_xai_on_image(img_path: str, model, transform, device, methods: Tuple[str, ...] = ("Raw", "Rollout", "Flow")) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """Executa todos os métodos XAI em uma imagem."""
    pil_img, x = load_image_as_tensor(img_path, transform, device)
    pred, conf, probs, attentions = forward_with_attentions(model, x)
    maps = {}
    if "Raw" in methods:
        raw_v = xai_raw_attention(attentions)
        raw_14 = vector_to_patchmap(raw_v)
        maps["Raw"] = upscale_to_image(raw_14)
    if "Rollout" in methods:
        roll_v = xai_attention_rollout(attentions)
        roll_14 = vector_to_patchmap(roll_v)
        maps["Rollout"] = upscale_to_image(roll_14)
    if "Flow" in methods:
        flow_v = xai_attention_flow(attentions)
        flow_14 = vector_to_patchmap(flow_v)
        maps["Flow"] = upscale_to_image(flow_14)
    return pil_img, pred, conf, maps