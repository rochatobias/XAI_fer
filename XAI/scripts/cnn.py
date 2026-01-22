"""CNN XAI - Métodos de Explicabilidade para ConvNeXt (CAM-based)."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Dict, Optional

import timm
from timm.data import resolve_model_data_config, create_transform

from config import (
    CNN_MODEL_PATH, CNN_MODEL_NAME, CNN_NUM_CLASSES,
    CNN_XAI_METHODS, IMG_SIZE, get_device
)

EPS = 1e-12


def _strip_module_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """Remove prefixo 'module.' do state_dict se existir (para modelos treinados com DataParallel)."""
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Normaliza heatmap para [0, 1]."""
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm)
    hm = hm - hm.min()
    denom = hm.max() + EPS
    return hm / denom


def _resize_heatmap(hm: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Redimensiona heatmap para o tamanho da imagem usando PIL."""
    hm = _normalize_heatmap(hm)
    # Converte para PIL Image para resize
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_pil = Image.fromarray(hm_uint8)
    hm_resized = hm_pil.resize(size, resample=Image.BICUBIC)
    # Converte de volta para array normalizado
    hm_resized = np.array(hm_resized, dtype=np.float32) / 255.0
    return _normalize_heatmap(hm_resized)


def load_cnn_model(
    model_path: str = CNN_MODEL_PATH,
    model_name: str = CNN_MODEL_NAME,
    num_classes: int = CNN_NUM_CLASSES,
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, dict, torch.device]:
    """
    Carrega o modelo CNN (ConvNeXt) para classificação.
    
    Returns:
        model: Modelo carregado em modo eval
        data_config: Configuração de pré-processamento do timm
        device: Device utilizado
    """
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    # Cria modelo
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model = model.to(device)
    
    # Carrega checkpoint
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Obtém config de pré-processamento
    temp_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    data_config = resolve_model_data_config(temp_model)
    del temp_model
    torch.cuda.empty_cache()
    
    print(f"CNN Model: {model_name} | num_classes: {num_classes}")
    print(f"Device: {device}")
    print(f"Checkpoint: {model_path}")
    
    return model, data_config, device


def build_cnn_transform(data_config: dict):
    """Constrói transformações de pré-processamento baseadas na config do modelo."""
    transform = create_transform(**data_config, is_training=False)
    img_size = (data_config["input_size"][2], data_config["input_size"][1])
    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])
    return transform, img_size, mean, std


def load_image_as_tensor_cnn(
    image_path: str,
    transform,
    device: torch.device
) -> Tuple[Image.Image, torch.Tensor]:
    """Carrega uma imagem e aplica as transformações para CNN."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return img, x


@torch.no_grad()
def forward_cnn(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    """
    Forward pass no modelo CNN.
    
    Returns:
        pred_idx: Índice da classe predita
        conf: Confiança da predição
        probs: Array de probabilidades
    """
    logits = model(x)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_idx = int(probs.argmax().item())
    conf = float(probs[pred_idx].item())
    return pred_idx, conf, probs.cpu().numpy()


def _get_target_layers(model: torch.nn.Module) -> list:
    """Obtém as camadas alvo para CAM no ConvNeXt."""
    # ConvNeXt: stages[-1].blocks[-1].conv_dw é a camada convolucional depthwise
    return [model.stages[-1].blocks[-1].conv_dw]


def _cam_map(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    method: str,
    target_layers: list
) -> np.ndarray:
    """Gera mapa CAM usando o método especificado."""
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


def run_xai_on_image_cnn(
    img_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    methods: Tuple[str, ...] = None
) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """
    Executa todos os métodos XAI CAM-based em uma imagem.
    
    Args:
        img_path: Caminho da imagem
        model: Modelo CNN carregado
        transform: Transformações de pré-processamento
        device: Device
        methods: Tupla de métodos a executar (default: CNN_XAI_METHODS)
    
    Returns:
        pil_img: Imagem PIL original
        pred_idx: Índice da classe predita
        conf: Confiança da predição
        maps: Dicionário {método: heatmap_224x224}
    """
    if methods is None:
        methods = tuple(CNN_XAI_METHODS)
    
    pil_img, x = load_image_as_tensor_cnn(img_path, transform, device)
    pred_idx, conf, probs = forward_cnn(model, x)
    
    target_layers = _get_target_layers(model)
    maps = {}
    
    for method in methods:
        # Normaliza nome do método
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


def plot_cnn_xai_overlays(
    pil_img: Image.Image,
    maps_dict: Dict[str, np.ndarray],
    title_prefix: str = "CNN XAI",
    save_path: Optional[str] = None,
    show: bool = True,
    overlay_alpha: float = 0.35
):
    """Plota imagem original com overlays dos mapas XAI."""
    import matplotlib.pyplot as plt
    
    img = pil_img.resize(IMG_SIZE)
    img_np = np.array(img)
    
    n = len(maps_dict)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title(f"{title_prefix} - Original")
    
    for i, (name, m224) in enumerate(maps_dict.items(), 1):
        axes[i].imshow(img_np)
        axes[i].imshow(m224, alpha=overlay_alpha, cmap='jet')
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