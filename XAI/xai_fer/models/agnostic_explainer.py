"""
Explicador XAI agnóstico (LIME e SHAP).

Métodos que funcionam como caixa-preta: perturbam a entrada e observam a
mudança na saída, sem acesso a gradientes nem atenções internas.

1. **LIME** (Ribeiro et al., 2016) — segmenta a imagem em superpixels
   (SLIC), gera vizinhança perturbada ligando/desligando superpixels
   e ajusta um modelo linear local. Os pesos dos superpixels são usados
   como heatmap contínuo (apenas contribuições positivas).
2. **SHAP** (Lundberg & Lee, 2017) via ``shap.Explainer`` com algoritmo
   ``partition`` — calcula valores de Shapley aproximados por particionamento
   hierárquico. Mais lento que LIME mas teoricamente mais fundamentado.
   Usa masker de blur (128 × 128) para ocluir regiões.

Custo: ~10-100× mais lentos que métodos nativos (atenção, CAM).
Por isso, são aplicados apenas ao subconjunto estratificado.

Ferramentas utilizadas:
    - ``lime`` (lime_image) + ``scikit-image`` (SLIC) para LIME.
    - ``shap`` (partition explainer) para SHAP.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Optional, Tuple

from xai_fer.config import IMG_SIZE, MEAN, STD, N_SAMPLES_AGNOSTIC, get_device

EPS = 1e-12

# Suprimir warnings do SHAP/tqdm
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ==============================================================================
# Helpers
# ==============================================================================


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Normaliza heatmap para [0, 1]."""
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm)
    hm = hm - hm.min()
    return hm / (hm.max() + EPS)


# ==============================================================================
# Model Wrapper
# ==============================================================================


class ModelWrapper:
    """Wrapper para adaptar modelos ViT/CNN à interface esperada por LIME/SHAP.

    LIME e SHAP esperam uma função ``f(batch_images) → probabilities`` onde
    ``batch_images`` é um array numpy [N, H, W, C] em [0, 255].
    Este wrapper cuida de: re-escalar, transpor, normalizar e despachar
    para o modelo correto (ViT usa ``pixel_values``, CNN usa forward direto).

    Args:
        model: Modelo PyTorch (ViT ou CNN).
        device: Device do modelo.
        model_type: ``'vit'`` ou ``'cnn'``.
        mean: Média de normalização ImageNet.
        std: Desvio padrão de normalização ImageNet.
        img_size: Tamanho esperado da entrada.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_type: str = "vit",
        mean: Tuple[float, ...] = MEAN,
        std: Tuple[float, ...] = STD,
        img_size: Tuple[int, int] = IMG_SIZE,
    ):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.img_size = img_size
        self.model.eval()

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Prediz probabilidades para batch de imagens.

        Args:
            images: [N, H, W, C] ou [H, W, C] em [0, 255] ou [0, 1].

        Returns:
            Array [N, num_classes] de probabilidades.
        """
        with torch.no_grad():
            if images.ndim == 3:
                images = images[None, ...]

            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            else:
                images = images.astype(np.float32)

            x = torch.tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
            x = (x - self.mean) / (self.std + EPS)

            if self.model_type == "vit":
                logits = self.model(pixel_values=x).logits
            else:
                logits = self.model(x)

            return F.softmax(logits, dim=-1).cpu().numpy()

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
    num_samples: int = 2500,
    num_features: int = 20,
) -> Tuple[np.ndarray, Dict]:
    """Gera explicação LIME para uma imagem.

    Segmenta com SLIC (120 superpixels), pertuba e ajusta modelo linear local.
    O heatmap retornado contém apenas contribuições positivas (pesos dos
    superpixels mais relevantes), normalizado para [0, 1].

    Args:
        model: Modelo PyTorch.
        pil_img: Imagem PIL.
        device: Device.
        model_type: ``'vit'`` ou ``'cnn'``.
        target_class: Classe alvo (``None`` = usa top prediction).
        num_samples: Número de perturbações para o modelo linear local.
        num_features: Número de superpixels top-positivos a incluir.

    Returns:
        Tupla ``(heatmap_224x224, info_dict)``.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import slic
    except ImportError:
        raise ImportError("LIME não instalado. Execute: pip install lime scikit-image")

    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)

    wrapper = ModelWrapper(model, device, model_type)
    explainer = lime_image.LimeImageExplainer()

    def slic_segmentation(image):
        return slic(image, n_segments=120, compactness=10, sigma=1, start_label=0)

    explanation = explainer.explain_instance(
        img_array,
        wrapper.predict_proba,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=slic_segmentation,
    )

    if target_class is None:
        target_class = explanation.top_labels[0]

    segments = explanation.segments
    sp_weights = dict(explanation.local_exp[target_class])

    heatmap = np.zeros_like(segments, dtype=np.float32)
    pos = [(sp, w) for sp, w in sp_weights.items() if w > 0]
    pos = sorted(pos, key=lambda t: t[1], reverse=True)[:num_features]

    if len(pos) > 0:
        ws = np.array([w for _, w in pos], dtype=np.float32)
        wmin, wmax = float(ws.min()), float(ws.max())
        scale = (wmax - wmin) + EPS
        for sp_id, w in pos:
            w01 = (float(w) - wmin) / scale
            heatmap[segments == sp_id] = w01

    heatmap = _normalize_heatmap(heatmap)

    info = {
        "target_class": target_class,
        "num_samples": num_samples,
        "num_features": num_features,
        "top_labels": list(explanation.top_labels),
        "note": "lime_superpixel_weight_heatmap_positive_only",
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
    max_evals: int = 1300,
    algorithm: str = "partition",
    masker: str = "blur(128,128)",
    positive_only: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Gera explicação SHAP (partition) para uma imagem.

    Usa ``shap.maskers.Image`` com blur para preencher regiões ocultadas.
    Por padrão retorna escala simétrica [-1, 1] (``positive_only=False``),
    adequada para visualização divergente (colormap ``RdBu_r``).

    Args:
        model: Modelo PyTorch.
        pil_img: Imagem PIL.
        device: Device.
        model_type: ``'vit'`` ou ``'cnn'``.
        target_class: Classe alvo (``None`` = usa top prediction).
        max_evals: Budget de avaliações do modelo.
        algorithm: Algoritmo shap (``'partition'`` é o mais viável).
        masker: Tipo de masker (``'blur(128,128)'``).
        positive_only: Se ``True``, mantém apenas contribuições positivas.

    Returns:
        Tupla ``(heatmap_224x224, info_dict)``.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP não instalado. Execute: pip install shap")

    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.uint8)

    wrapper = ModelWrapper(model, device, model_type)

    probs = wrapper.predict_proba(img_array[None, ...])[0]
    pred_idx = int(np.argmax(probs))
    if target_class is None:
        target_class = pred_idx

    img_masker = shap.maskers.Image(masker, img_array.shape)
    explainer = shap.Explainer(wrapper.predict_proba, img_masker, algorithm=algorithm)

    sv = explainer(img_array[None, ...], max_evals=max_evals, silent=True)
    vals = sv.values

    if vals.ndim == 5:
        class_vals = vals[0, :, :, :, target_class]
    elif vals.ndim == 4:
        class_vals = vals[0]
    else:
        raise RuntimeError(f"Formato inesperado de sv.values: {vals.shape}")

    heatmap = class_vals.sum(axis=-1).astype(np.float32)

    if positive_only:
        heatmap = np.maximum(heatmap, 0.0)
        heatmap = _normalize_heatmap(heatmap)
    else:
        abs_max = max(abs(heatmap.min()), abs(heatmap.max())) + EPS
        heatmap = heatmap / abs_max

    info = {
        "target_class": int(target_class),
        "pred_class": int(pred_idx),
        "confidence": float(probs[target_class]),
        "algorithm": algorithm,
        "masker": masker,
        "max_evals": int(max_evals),
        "positive_only": bool(positive_only),
        "note": "shap_partition_image",
    }
    return heatmap, info


# ==============================================================================
# Interface Unificada
# ==============================================================================


def run_agnostic_xai(
    img_path: str,
    model: torch.nn.Module,
    device: torch.device,
    model_type: str = "vit",
    methods: Tuple[str, ...] = ("LIME", "SHAP"),
) -> Tuple[Image.Image, int, float, Dict[str, np.ndarray]]:
    """Executa métodos XAI agnósticos em uma imagem.

    Args:
        img_path: Caminho da imagem.
        model: Modelo PyTorch.
        device: Device.
        model_type: ``'vit'`` ou ``'cnn'``.
        methods: Métodos a executar.

    Returns:
        Tupla ``(pil_img, pred_idx, confidence, maps_dict)``.
    """
    pil_img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)

    wrapper = ModelWrapper(model, device, model_type)
    probs = wrapper.predict_proba(np.array(pil_img))[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    maps: Dict[str, np.ndarray] = {}

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
