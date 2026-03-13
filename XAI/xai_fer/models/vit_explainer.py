"""
Explicador XAI para Vision Transformers (ViT).

Implementa três métodos de interpretabilidade baseados em mapas de atenção:

1. **Raw Attention** — média das heads da última camada; rápido mas ruidoso.
2. **Attention Rollout** (Abnar & Zuidema, 2020) — multiplica matrizes de
   atenção camada a camada, acumulando a influência do token [CLS] sobre
   cada patch. Usa peso residual para simular a conexão skip do Transformer.
3. **Attention Flow** — propaga massa do [CLS] por todas as camadas, com
   exponenciação opcional (``power > 1``) para destacar conexões fortes.

Cada método retorna um heatmap 224 × 224 normalizado em [0, 1], pronto para
overlay sobre a imagem original.

Ferramentas utilizadas:
    - HuggingFace Transformers (``AutoModelForImageClassification``) para
      carregar o checkpoint ViT fine-tuned.
    - timm apenas para obter os parâmetros de pré-processamento do ConvNeXt
      (``resolve_model_data_config``), garantindo compatibilidade entre modelos.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoConfig
import timm
from timm.data import resolve_model_data_config
from typing import Dict, Optional, Tuple

from xai_fer.config import (
    VIT_MODEL_DIR,
    ROLLOUT_RESIDUAL_WEIGHT,
    FLOW_RESIDUAL_WEIGHT,
    FLOW_POWER,
    get_device,
)

EPS = 1e-12


# ==============================================================================
# Modelo
# ==============================================================================


def load_vit_model(
    model_dir: str = VIT_MODEL_DIR,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoConfig, torch.device]:
    """Carrega o modelo ViT com suporte a retorno de atenções.

    Args:
        model_dir: Diretório do checkpoint HuggingFace.
        device: Device explícito (``'cuda'``, ``'cpu'``). Se ``None``, auto-detect.

    Returns:
        Tupla ``(model, config, device)``.
    """
    dev = torch.device(device) if device else get_device()
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    ).to(dev)
    model.config.output_attentions = True
    model.eval()
    print(f"ViT carregado | type={cfg.model_type} | labels={getattr(cfg, 'num_labels', None)} | device={dev}")
    return model, cfg, dev


def build_transform_from_convnext() -> Tuple[transforms.Compose, tuple, tuple, tuple]:
    """Constrói transformações de pré-processamento baseadas no ConvNeXt.

    Usa um modelo temporário do timm apenas para extrair ``mean``, ``std``
    e ``input_size``, garantindo paridade com o pipeline de treinamento.

    Returns:
        Tupla ``(transform, size, mean, std)``.
    """
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
    print(f"Transform ViT: size={size}, mean={mean}, std={std}")
    return tfm, size, mean, std


# ==============================================================================
# Helpers internos
# ==============================================================================


def _avg_heads(attn_layer: torch.Tensor) -> torch.Tensor:
    """Média das atenções sobre todas as heads de uma camada."""
    return attn_layer[0].mean(dim=0)


def _cls_to_patch_map(attn_TT: torch.Tensor) -> torch.Tensor:
    """Extrai atenção do token [CLS] para os patches (exclui o próprio CLS)."""
    return attn_TT[0, 1:]


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normaliza array para [0, 1]."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


# ==============================================================================
# Forward
# ==============================================================================


def load_image_as_tensor(
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[Image.Image, torch.Tensor]:
    """Carrega imagem do disco e aplica transform para tensor."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return img, x


@torch.no_grad()
def forward_with_attentions(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> Tuple[int, float, np.ndarray, tuple]:
    """Forward pass que retorna predição, confiança e mapas de atenção.

    Returns:
        Tupla ``(pred_idx, confidence, probs_array, attentions_tuple)``.
    """
    out = model(pixel_values=x, output_attentions=True)
    logits = out.logits
    probs = F.softmax(logits, dim=-1)
    pred = int(probs.argmax(dim=-1).item())
    conf = float(probs.max(dim=-1).values.item())
    attentions = out.attentions
    return pred, conf, probs.squeeze(0).cpu().numpy(), attentions


# ==============================================================================
# Métodos XAI
# ==============================================================================


def xai_raw_attention(attentions: tuple) -> np.ndarray:
    """Raw Attention: média das heads da última camada.

    Método mais simples — mostra a atenção direta do token [CLS] para cada
    patch na última camada. Tende a ser ruidoso pois ignora camadas anteriores.

    Args:
        attentions: Tupla de tensores de atenção (uma por camada).

    Returns:
        Vetor 1D com 196 valores (grid 14 × 14).
    """
    last = attentions[-1]
    A = _avg_heads(last)
    v = _cls_to_patch_map(A)
    return v.detach().cpu().numpy()


def xai_attention_rollout(
    attentions: tuple,
    residual_weight: float = ROLLOUT_RESIDUAL_WEIGHT,
) -> np.ndarray:
    """Attention Rollout — composição multiplicativa camada a camada.

    Acumula ``R = R @ A'`` onde ``A' = w·A + (1-w)·I`` (peso residual
    simula a skip connection). Resultado final é a linha [CLS] da
    matriz acumulada.

    Args:
        attentions: Tupla de tensores de atenção.
        residual_weight: Peso da atenção vs. identidade (skip connection).

    Returns:
        Vetor normalizado de 196 valores (grid 14 × 14).
    """
    L = len(attentions)
    A0 = _avg_heads(attentions[0])
    T = A0.shape[-1]
    I = torch.eye(T, device=A0.device, dtype=A0.dtype)

    R = I
    for layer_idx in range(L):
        A = _avg_heads(attentions[layer_idx])
        A = residual_weight * A + (1.0 - residual_weight) * I
        A = A / (A.sum(dim=-1, keepdim=True) + EPS)
        R = R @ A

    v = R[0, 1:]
    v = v / (v.sum() + EPS)
    return v.detach().cpu().numpy()


def xai_attention_flow(
    attentions: tuple,
    residual_weight: float = FLOW_RESIDUAL_WEIGHT,
    power: float = FLOW_POWER,
) -> np.ndarray:
    """Attention Flow — propagação de massa do [CLS] por todas as camadas.

    Começa com massa 1.0 no token [CLS] e propaga via ``f = f @ A'``
    onde ``A'`` é a atenção com exponenciação (``power > 1`` destaca
    conexões fortes) e peso residual.

    Args:
        attentions: Tupla de tensores de atenção.
        residual_weight: Peso da atenção vs. identidade.
        power: Expoente aplicado à atenção antes da propagação.

    Returns:
        Vetor normalizado de 196 valores (grid 14 × 14).
    """
    L = len(attentions)
    A0 = _avg_heads(attentions[0])
    T = A0.shape[-1]
    I = torch.eye(T, device=A0.device, dtype=A0.dtype)

    f = torch.zeros(T, device=A0.device, dtype=A0.dtype)
    f[0] = 1.0  # massa no CLS

    for layer_idx in range(L):
        A = _avg_heads(attentions[layer_idx])
        A = torch.pow(A, power)
        A = residual_weight * A + (1.0 - residual_weight) * I
        A = A / (A.sum(dim=-1, keepdim=True) + EPS)
        f = f @ A

    v = f[1:]
    v = v / (v.sum() + EPS)
    return v.detach().cpu().numpy()


# ==============================================================================
# Pós-processamento
# ==============================================================================


def vector_to_patchmap(
    v_196: np.ndarray,
    grid_size: int = 14,
    p: float = 99.0,
) -> np.ndarray:
    """Converte vetor 196D em mapa 14 × 14 com clipping percentílico."""
    m = v_196.reshape(grid_size, grid_size).astype(np.float32)
    m = m - m.min()
    denom = np.percentile(m, p) + 1e-8
    m = np.clip(m / denom, 0, 1)
    return m


def upscale_to_image(
    map_14: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Aumenta resolução do mapa 14 × 14 para tamanho da imagem."""
    m = (map_14 * 255).astype(np.uint8)
    m_img = Image.fromarray(m).resize(out_size, resample=Image.BILINEAR)
    return np.array(m_img) / 255.0


# ==============================================================================
# Interface pública
# ==============================================================================


def run_xai_on_image(
    img_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    methods: Tuple[str, ...] = ("Raw", "Rollout", "Flow"),
) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """Executa todos os métodos XAI de atenção em uma imagem.

    Args:
        img_path: Caminho da imagem no disco.
        model: Modelo ViT carregado (com ``output_attentions=True``).
        transform: Pipeline de pré-processamento.
        device: Device do modelo.
        methods: Tupla com nomes dos métodos a executar.

    Returns:
        Tupla ``(pil_img, pred_idx, confidence, maps_dict)`` onde
        ``maps_dict`` mapeia nome → heatmap 224 × 224.
    """
    pil_img, x = load_image_as_tensor(img_path, transform, device)
    pred, conf, _probs, attentions = forward_with_attentions(model, x)

    maps: Dict[str, np.ndarray] = {}
    if "Raw" in methods:
        maps["Raw"] = upscale_to_image(vector_to_patchmap(xai_raw_attention(attentions)))
    if "Rollout" in methods:
        maps["Rollout"] = upscale_to_image(vector_to_patchmap(xai_attention_rollout(attentions)))
    if "Flow" in methods:
        maps["Flow"] = upscale_to_image(vector_to_patchmap(xai_attention_flow(attentions)))

    return pil_img, pred, conf, maps
