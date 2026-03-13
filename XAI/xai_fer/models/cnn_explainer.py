"""
Explicador XAI para CNN (ConvNeXt) via métodos baseados em CAM.

Implementa três métodos de Class Activation Mapping:

1. **GradCAM** (Selvaraju et al., 2017) — pondera os feature maps da última
   camada convolucional pelos gradientes médios da classe predita.  É o
   método padrão em XAI para CNNs, rápido e amplamente aceito.
2. **GradCAM++** (Chattopadhay et al., 2018) — extensão que usa pesos de
   segunda ordem para lidar melhor com múltiplas ocorrências do objeto.
3. **LayerCAM** (Jiang et al., 2021) — combina gradientes positivos com
   mapas de ativação; oferece bom equilíbrio velocidade/qualidade.
   (ScoreCAM foi descartado por ser ~10× mais lento sem ganho proporcional.)

O modelo utilizado é o **ConvNeXt Base** (timm), fine-tuned nas 7 classes
do FER. A camada alvo para CAM é ``stages[-1].blocks[-1].conv_dw``
(última depthwise convolution do último estágio).

Ferramentas utilizadas:
    - timm para instanciar o ConvNeXt e obter parâmetros de data config.
    - pytorch-grad-cam para GradCAM, GradCAM++, LayerCAM.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Optional, Tuple

import timm
from timm.data import resolve_model_data_config, create_transform

from xai_fer.config import (
    CNN_MODEL_PATH,
    CNN_MODEL_NAME,
    CNN_NUM_CLASSES,
    CNN_XAI_METHODS,
    IMG_SIZE,
    get_device,
)

EPS = 1e-12


# ==============================================================================
# Helpers internos
# ==============================================================================


def _strip_module_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """Remove prefixo ``module.`` de state_dicts treinados com DataParallel."""
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Normaliza heatmap para [0, 1] com tratamento de NaN."""
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm)
    hm = hm - hm.min()
    return hm / (hm.max() + EPS)


def _resize_heatmap(hm: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Redimensiona heatmap via interpolação bicúbica (PIL) e renormaliza."""
    hm = _normalize_heatmap(hm)
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_pil = Image.fromarray(hm_uint8)
    hm_resized = hm_pil.resize(size, resample=Image.BICUBIC)
    hm_resized = np.array(hm_resized, dtype=np.float32) / 255.0
    return _normalize_heatmap(hm_resized)


# ==============================================================================
# Modelo
# ==============================================================================


def load_cnn_model(
    model_path: str = CNN_MODEL_PATH,
    model_name: str = CNN_MODEL_NAME,
    num_classes: int = CNN_NUM_CLASSES,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict, torch.device]:
    """Carrega o modelo ConvNeXt fine-tuned.

    Args:
        model_path: Caminho do arquivo ``.pth`` com pesos.
        model_name: Nome do modelo timm (``convnext_base``).
        num_classes: Número de classes de saída (7 emoções).
        device: Device explícito; se ``None``, auto-detect.

    Returns:
        Tupla ``(model, data_config, device)``.
    """
    dev = torch.device(device) if device else get_device()

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model = model.to(dev)

    state_dict = torch.load(model_path, map_location=dev, weights_only=True)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Obtém config de pré-processamento
    temp_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    data_config = resolve_model_data_config(temp_model)
    del temp_model
    torch.cuda.empty_cache()

    print(f"CNN carregado | model={model_name} | classes={num_classes} | device={dev}")
    return model, data_config, dev


def build_cnn_transform(
    data_config: dict,
) -> Tuple:
    """Constrói transformações de pré-processamento partir do data_config do timm.

    Returns:
        Tupla ``(transform, img_size, mean, std)``.
    """
    transform = create_transform(**data_config, is_training=False)
    img_size = (data_config["input_size"][2], data_config["input_size"][1])
    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])
    return transform, img_size, mean, std


# ==============================================================================
# Forward
# ==============================================================================


def load_image_as_tensor_cnn(
    image_path: str,
    transform,
    device: torch.device,
) -> Tuple[Image.Image, torch.Tensor]:
    """Carrega imagem e aplica transform para tensor CNN."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return img, x


@torch.no_grad()
def forward_cnn(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> Tuple[int, float, np.ndarray]:
    """Forward pass no modelo CNN.

    Returns:
        Tupla ``(pred_idx, confidence, probs_array)``.
    """
    logits = model(x)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_idx = int(probs.argmax().item())
    conf = float(probs[pred_idx].item())
    return pred_idx, conf, probs.cpu().numpy()


# ==============================================================================
# CAM
# ==============================================================================


def _get_target_layers(model: torch.nn.Module) -> list:
    """Retorna a camada alvo para CAM no ConvNeXt.

    Usa a última depthwise convolution (``stages[-1].blocks[-1].conv_dw``).
    """
    return [model.stages[-1].blocks[-1].conv_dw]


def _cam_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    method: str,
    target_layers: list,
) -> np.ndarray:
    """Gera mapa CAM para a classe alvo usando o método especificado.

    Args:
        model: Modelo CNN.
        x: Tensor de entrada ``[1, 3, H, W]``.
        target_class: Índice da classe para gerar a explicação.
        method: Nome do método (``'gradcam'``, ``'gradcam++'``, ``'layercam'``).
        target_layers: Lista de camadas alvo do modelo.

    Returns:
        Heatmap 224 × 224 normalizado em [0, 1].
    """
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(target_class)]

    cam_classes = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "layercam": LayerCAM,
        "scorecam": ScoreCAM,
    }

    method_lower = method.lower()
    if method_lower not in cam_classes:
        raise ValueError(f"Método inválido: {method}. Opções: {list(cam_classes.keys())}")

    cam = cam_classes[method_lower](model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]
    return _resize_heatmap(grayscale_cam, size=IMG_SIZE)


# ==============================================================================
# Interface pública
# ==============================================================================


def run_xai_on_image_cnn(
    img_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    methods: Tuple[str, ...] = None,
) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """Executa todos os métodos CAM-based em uma imagem.

    Normaliza nomes de métodos para compatibilidade (ex: ``'GradCAM++'``
    → ``'gradcam++'``).

    Args:
        img_path: Caminho da imagem.
        model: Modelo CNN carregado.
        transform: Pipeline de pré-processamento.
        device: Device.
        methods: Tupla de métodos a executar (default: ``CNN_XAI_METHODS``).

    Returns:
        Tupla ``(pil_img, pred_idx, confidence, maps_dict)``.
    """
    if methods is None:
        methods = tuple(CNN_XAI_METHODS)

    pil_img, x = load_image_as_tensor_cnn(img_path, transform, device)
    pred_idx, conf, _probs = forward_cnn(model, x)

    target_layers = _get_target_layers(model)
    maps: Dict[str, np.ndarray] = {}

    for method in methods:
        method_key = method
        method_lower = method.lower().replace("++", "pp").replace("-", "")

        if "gradcampp" in method_lower or "gradcam++" in method.lower():
            cam_method = "gradcam++"
        elif "gradcam" in method_lower:
            cam_method = "gradcam"
        elif "layercam" in method_lower:
            cam_method = "layercam"
        elif "scorecam" in method_lower:
            cam_method = "scorecam"
        else:
            cam_method = method.lower()

        maps[method_key] = _cam_map(model, x, pred_idx, cam_method, target_layers)

    return pil_img, pred_idx, conf, maps
