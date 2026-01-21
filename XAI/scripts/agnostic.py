# ==============================================================================
# Agnostic XAI - Métodos Model-Agnostic (LIME e SHAP)
# ==============================================================================
# Estes métodos são mais lentos mas funcionam com qualquer modelo.
# Use N_SAMPLES_AGNOSTIC separado para controlar quantas imagens processar.
# ==============================================================================

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Dict, Optional, Callable
import warnings

from config import IMG_SIZE, MEAN, STD, N_SAMPLES_AGNOSTIC, get_device

EPS = 1e-12


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Normaliza heatmap para [0, 1]."""
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm)
    hm = hm - hm.min()
    denom = hm.max() + EPS
    return hm / denom


class ModelWrapper:
    """
    Wrapper para adaptar modelos ViT/CNN para interfaces LIME/SHAP.
    
    LIME e SHAP esperam funções que recebem arrays numpy e retornam probabilidades.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_type: str = "vit",
        mean: Tuple[float, ...] = MEAN,
        std: Tuple[float, ...] = STD,
        img_size: Tuple[int, int] = IMG_SIZE
    ):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.img_size = img_size
        self.model.eval()
    
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """
        Prediz probabilidades para batch de imagens.
        
        Args:
            images: Array de imagens [N, H, W, C] em [0, 255] ou [0, 1]
        
        Returns:
            probs: Array [N, num_classes] de probabilidades
        """
        with torch.no_grad():
            # Converte para tensor
            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            
            # [N, H, W, C] -> [N, C, H, W]
            x = torch.tensor(images, dtype=torch.float32, device=self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)
            
            # Normaliza
            x = (x - self.mean) / (self.std + EPS)
            
            # Forward
            if self.model_type == "vit":
                logits = self.model(pixel_values=x).logits
            else:
                logits = self.model(x)
            
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            return probs
    
    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images)


# ==============================================================================
# LIME
# ==============================================================================

def lime_explain(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: torch.device,
    model_type: str = "vit",
    target_class: Optional[int] = None,
    num_samples: int = 1000,
    num_features: int = 100
) -> Tuple[np.ndarray, Dict]:
    """
    Gera explicação LIME para uma imagem.
    
    Args:
        model: Modelo PyTorch
        pil_img: Imagem PIL
        device: Device
        model_type: 'vit' ou 'cnn'
        target_class: Classe alvo (None = classe predita)
        num_samples: Número de perturbações para LIME
        num_features: Número de superpixels a mostrar
    
    Returns:
        heatmap: Heatmap 224x224 normalizado
        info: Dicionário com informações adicionais
    """
    try:
        from lime import lime_image
        from skimage.segmentation import quickshift
    except ImportError:
        raise ImportError("LIME não instalado. Execute: pip install lime")
    
    # Prepara imagem
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)  # [H, W, C] em [0, 255]
    
    # Wrapper do modelo
    wrapper = ModelWrapper(model, device, model_type)
    
    # Cria explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Gera explicação
    explanation = explainer.explain_instance(
        img_array,
        wrapper.predict_proba,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
    )
    
    # Determina classe alvo
    if target_class is None:
        target_class = explanation.top_labels[0]
    
    # Obtém máscara de importância
    _, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )
    
    # Converte máscara para heatmap contínuo
    segments = explanation.segments
    local_exp = dict(explanation.local_exp[target_class])
    
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for segment_id, weight in local_exp.items():
        heatmap[segments == segment_id] = weight
    
    # Normaliza
    heatmap = _normalize_heatmap(heatmap)
    
    info = {
        "target_class": target_class,
        "num_samples": num_samples,
        "num_features": num_features,
        "top_labels": explanation.top_labels
    }
    
    return heatmap, info


# ==============================================================================
# SHAP
# ==============================================================================

def shap_explain(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: torch.device,
    model_type: str = "vit",
    target_class: Optional[int] = None,
    background_samples: int = 50,
    method: str = "gradient"
) -> Tuple[np.ndarray, Dict]:
    """
    Gera explicação SHAP para uma imagem.
    
    Args:
        model: Modelo PyTorch
        pil_img: Imagem PIL
        device: Device
        model_type: 'vit' ou 'cnn'
        target_class: Classe alvo (None = classe predita)
        background_samples: Número de amostras de background
        method: 'gradient' ou 'deep' (GradientExplainer ou DeepExplainer)
    
    Returns:
        heatmap: Heatmap 224x224 normalizado
        info: Dicionário com informações adicionais
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP não instalado. Execute: pip install shap")
    
    # Prepara imagem
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0  # [H, W, C] em [0, 1]
    
    # Tensor normalizado
    x = torch.tensor(img_array, dtype=torch.float32, device=device)
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean_t) / (std_t + EPS)
    
    # Determina classe alvo se não especificada
    model.eval()
    with torch.no_grad():
        if model_type == "vit":
            logits = model(pixel_values=x_norm).logits
        else:
            logits = model(x_norm)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        if target_class is None:
            target_class = int(probs.argmax().item())
    
    # Gera background (imagens pretas ou ruído)
    background = torch.zeros((background_samples, 3, *IMG_SIZE), device=device)
    
    # Cria função que aceita tensor e retorna logits da classe alvo
    def model_fn(x_input):
        if model_type == "vit":
            return model(pixel_values=x_input).logits
        else:
            return model(x_input)
    
    try:
        # Tenta GradientExplainer (mais rápido)
        if method == "gradient":
            explainer = shap.GradientExplainer(model_fn, background)
            shap_values = explainer.shap_values(x_norm)
        else:
            # DeepExplainer
            explainer = shap.DeepExplainer(model_fn, background)
            shap_values = explainer.shap_values(x_norm)
        
        # shap_values é [num_classes, batch, C, H, W] ou [batch, C, H, W]
        if isinstance(shap_values, list):
            # Multi-output: pega classe alvo
            sv = shap_values[target_class][0]  # [C, H, W]
        else:
            sv = shap_values[0]  # [C, H, W]
        
        # Agrega canais (soma absoluta)
        heatmap = np.abs(sv).sum(axis=0)  # [H, W]
        heatmap = _normalize_heatmap(heatmap)
        
        info = {
            "target_class": target_class,
            "method": method,
            "background_samples": background_samples,
            "confidence": float(probs[target_class].item())
        }
        
        return heatmap, info
        
    except Exception as e:
        warnings.warn(f"SHAP falhou: {e}. Retornando heatmap vazio.")
        heatmap = np.zeros(IMG_SIZE, dtype=np.float32)
        info = {"error": str(e), "target_class": target_class}
        return heatmap, info


# ==============================================================================
# Interface Unificada
# ==============================================================================

def run_agnostic_xai(
    img_path: str,
    model: torch.nn.Module,
    device: torch.device,
    model_type: str = "vit",
    methods: Tuple[str, ...] = ("LIME", "SHAP")
) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """
    Executa métodos XAI agnósticos (LIME/SHAP) em uma imagem.
    
    Args:
        img_path: Caminho da imagem
        model: Modelo PyTorch
        device: Device
        model_type: 'vit' ou 'cnn'
        methods: Métodos a executar
    
    Returns:
        pil_img: Imagem PIL
        pred_idx: Classe predita
        conf: Confiança
        maps: Dicionário {método: heatmap_224x224}
    """
    pil_img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    
    # Predição inicial
    wrapper = ModelWrapper(model, device, model_type)
    probs = wrapper.predict_proba(np.array(pil_img))[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    
    maps = {}
    
    for method in methods:
        method_lower = method.lower()
        try:
            if method_lower == "lime":
                hm, _ = lime_explain(model, pil_img, device, model_type, target_class=pred_idx)
                maps["LIME"] = hm
            elif method_lower == "shap":
                hm, _ = shap_explain(model, pil_img, device, model_type, target_class=pred_idx)
                maps["SHAP"] = hm
            else:
                warnings.warn(f"Método desconhecido: {method}")
        except Exception as e:
            warnings.warn(f"Erro em {method}: {e}")
            maps[method] = np.zeros(IMG_SIZE, dtype=np.float32)
    
    return pil_img, pred_idx, conf, maps
