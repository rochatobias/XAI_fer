"""Agnostic XAI - Métodos Model-Agnostic (LIME e SHAP).

Métodos mais lentos mas funcionam com qualquer modelo.
Use N_SAMPLES_AGNOSTIC para controlar o volume.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Dict, Optional
import warnings

from config import IMG_SIZE, MEAN, STD, N_SAMPLES_AGNOSTIC, get_device

EPS = 1e-12

# Suprimir warnings do SHAP/tqdm para evitar poluição do console
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


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
            # Aceita [H,W,C] ou [N,H,W,C]
            if images.ndim == 3:
                images = images[None, ...]

            # Converte faixa para [0,1]
            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            else:
                images = images.astype(np.float32)

            # [N, H, W, C] -> [N, C, H, W]
            x = torch.tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

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
    num_samples: int = 2500,
    num_features: int = 20
) -> Tuple[np.ndarray, Dict]:
    """
    Gera explicação LIME para uma imagem.

    Retorna um heatmap contínuo (pesos por superpixel), melhor do que máscara binária.
    Visual recomendado com colormap "Greens".
    """
    try:
        from lime import lime_image
        from skimage.segmentation import slic
    except ImportError:
        raise ImportError("LIME não instalado. Execute: pip install lime scikit-image")

    # Prepara imagem
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)  # [H, W, C] em [0, 255]

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
        segmentation_fn=slic_segmentation
    )

    if target_class is None:
        target_class = explanation.top_labels[0]

    segments = explanation.segments  # [H, W]
    sp_weights = dict(explanation.local_exp[target_class])  # {superpixel_id: weight}

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
        "note": "lime_superpixel_weight_heatmap_positive_only"
    }
    return heatmap, info


# ==============================================================================
# SHAP (puro) - Shapley-based
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
    positive_only: bool = False  # False = padrão acadêmico (mostra pos/neg)
) -> Tuple[np.ndarray, Dict]:
    """
    Gera explicação SHAP (puro) para uma imagem usando a biblioteca shap.

    Observações importantes:
    - SHAP em imagem é caro. O 'partition' é o mais viável na prática.
    - 'masker' controla como regiões ocultadas são preenchidas (blur costuma funcionar bem).
    - Por padrão retornamos apenas contribuições positivas para melhorar legibilidade.

    Returns:
        heatmap: Heatmap 224x224 normalizado [0,1]
        info: Dicionário com informações adicionais
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP não instalado. Execute: pip install shap")

    # Prepara imagem
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.uint8)  # [H,W,C] em [0,255]

    wrapper = ModelWrapper(model, device, model_type)

    # Determina classe alvo (pela predição do modelo)
    probs = wrapper.predict_proba(img_array[None, ...])[0]
    pred_idx = int(np.argmax(probs))
    if target_class is None:
        target_class = pred_idx

    # Masker de imagem: blur é o mais estável
    # Ex: "blur(128,128)" ou "inpaint_telea"
    img_masker = shap.maskers.Image(masker, img_array.shape)

    # Explainer: algorithm="partition" é Shapley-based e muito mais rápido que kernel em imagem
    explainer = shap.Explainer(wrapper.predict_proba, img_masker, algorithm=algorithm)

    # Computa shap values para 1 imagem (silent=True evita barra de progresso interna atrasada)
    sv = explainer(img_array[None, ...], max_evals=max_evals, silent=True)

    # sv.values pode ser:
    # (1, H, W, C, num_classes)  ou  (1, H, W, C) se o explainer reduzir saída
    vals = sv.values

    # Seleciona a classe alvo
    if vals.ndim == 5:
        # [N,H,W,C,K]
        class_vals = vals[0, :, :, :, target_class]
    elif vals.ndim == 4:
        # [N,H,W,C] -> assume já é da classe explicada (menos comum)
        class_vals = vals[0]
    else:
        raise RuntimeError(f"Formato inesperado de sv.values: {vals.shape}")

    # Agrega canais -> [H,W]
    heatmap = class_vals.sum(axis=-1).astype(np.float32)

    # Opcional: apenas contribuições positivas
    if positive_only:
        heatmap = np.maximum(heatmap, 0.0)
        heatmap = _normalize_heatmap(heatmap)
    else:
        # Normalização simétrica para visualização divergente (RdBu_r)
        # Mantém positivos e negativos, normaliza para [-1, 1]
        abs_max = max(abs(heatmap.min()), abs(heatmap.max())) + EPS
        heatmap = heatmap / abs_max  # Agora em [-1, 1]

    info = {
        "target_class": int(target_class),
        "pred_class": int(pred_idx),
        "confidence": float(probs[target_class]),
        "algorithm": algorithm,
        "masker": masker,
        "max_evals": int(max_evals),
        "positive_only": bool(positive_only),
        "note": "shap_partition_image"
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
                # use N_SAMPLES_AGNOSTIC como "budget" padrão, se você quiser:
                # max_evals = max(200, int(N_SAMPLES_AGNOSTIC))
                hm, _ = shap_explain(model, pil_img, device, model_type, target_class=pred_idx)
                maps["SHAP"] = hm
            else:
                warnings.warn(f"Método desconhecido: {method}")
        except Exception as e:
            warnings.warn(f"Erro em {method}: {e}")
            maps[method] = np.zeros(IMG_SIZE, dtype=np.float32)

    return pil_img, pred_idx, conf, maps
