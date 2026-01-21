def build_cnn():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    return model

def strip_module_prefix(sd, prefix="module."):
    if isinstance(sd, dict) and any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

model = build_cnn().to(DEVICE)

state = torch.load(CNN_CKPT_PATH, map_location=DEVICE)
state = strip_module_prefix(state)
model.load_state_dict(state, strict=True)
model.eval()

# preprocess igual ao timm (bem parecido com o que você já fez no treino)
tmp = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
data_config = resolve_model_data_config(tmp)
del tmp

transform = create_transform(**data_config, is_training=False)

IMG_SIZE = (data_config["input_size"][2], data_config["input_size"][1])  # (W,H) 224x224
print("data_config:", data_config)
print("IMG_SIZE:", IMG_SIZE)

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def to_tensor_for_model(pil_img: Image.Image) -> torch.Tensor:
    # timm transform já faz resize/crop/normalize adequados
    x = transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,224,224)
    return x

def softmax_confidence(logits: torch.Tensor, idx: int) -> float:
    prob = F.softmax(logits, dim=1)[0, idx].item()
    return float(prob)

def norm_heatmap_01(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm)
    hm = hm - hm.min()
    denom = hm.max() + 1e-12
    hm = hm / denom
    return hm

def resize_hm_to_224(hm: np.ndarray, size=(224,224)) -> np.ndarray:
    hm = norm_heatmap_01(hm)
    hm_rs = cv2.resize(hm, size, interpolation=cv2.INTER_CUBIC)
    return norm_heatmap_01(hm_rs)

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Target layer para ConvNeXt (timm): pegar uma camada conv confiável
# ConvNeXt tem stages[-1].blocks[-1].conv_dw (depthwise conv) -> ótimo p/ CAM
target_layers = [model.stages[-1].blocks[-1].conv_dw]

def cam_map_for_image(pil_img: Image.Image, target_class: int, method: str) -> np.ndarray:
    x = to_tensor_for_model(pil_img)

    targets = [ClassifierOutputTarget(target_class)]

    if method == "gradcam":
        cam = GradCAM(model=model, target_layers=target_layers)
    elif method == "gradcampp":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif method == "layercam":
        cam = LayerCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError("method inválido")

    # retorna (1, H, W) normalizado pela lib; ainda vamos padronizar
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]
    cam_map = resize_hm_to_224(grayscale_cam, size=(224,224))
    return cam_map

def scorecam_map_for_image(pil_img: Image.Image, target_class: int) -> np.ndarray:
    x = to_tensor_for_model(pil_img)
    targets = [ClassifierOutputTarget(target_class)]

    cam = ScoreCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H,W)

    cam_map = resize_hm_to_224(grayscale_cam, size=(224,224))
    return cam_map

base_tfm_01 = lambda pil: (torch.from_numpy(np.array(pil.resize((224,224)), dtype=np.uint8)).permute(2,0,1).float()/255.0)

def perturb_topk(x01: torch.Tensor, heatmap_224: np.ndarray, frac: float, mode="mean"):
    """
    x01: (3,224,224) em [0,1]
    heatmap_224: (224,224) em [0,1]
    frac: fração de pixels a perturbar
    """
    C,H,W = x01.shape
    hm = torch.tensor(heatmap_224, device=x01.device, dtype=torch.float32).view(-1)
    k = int(frac * H * W)
    if k <= 0:
        return x01.clone()

    topk_idx = torch.topk(hm, k=k, largest=True).indices
    x_pert = x01.clone().view(C, -1)

    if mode == "mean":
        baseline = x01.mean(dim=(1,2), keepdim=True).view(C,1)
        x_pert[:, topk_idx] = baseline
    elif mode == "zero":
        x_pert[:, topk_idx] = 0.0
    else:
        raise ValueError("mode deve ser 'mean' ou 'zero'")

    return x_pert.view(C,H,W)

@torch.no_grad()
def compute_aopc_cnn(model, pil_img, heatmap_224,
                     steps=(0.1,0.2,0.3,0.5),
                     target="true", true_label_idx=None,
                     perturb_mode="zero"):
    """
    Mesmo espírito do ViT: queda média de confiança ao perturbar top-k pixels.
    Retorna dict com drops e aopc_mean.
    """
    pil_img = pil_img.convert("RGB").resize((224,224))

    # base pred
    x_in = to_tensor_for_model(pil_img)                 # (1,3,224,224) já normalizado
    logits = model(x_in)
    pred_idx = int(logits.argmax(dim=1).item())

    if target == "pred":
        target_idx = pred_idx
    elif target == "true":
        if true_label_idx is None:
            raise ValueError("true_label_idx obrigatório se target='true'")
        target_idx = int(true_label_idx)
    else:
        raise ValueError("target deve ser 'pred' ou 'true'")

    base_conf = softmax_confidence(logits, target_idx)

    # perturba no espaço [0,1] e depois aplica normalização do timm
    x01 = base_tfm_01(pil_img).to(DEVICE)  # (3,224,224) [0,1]

    confs, drops = [], []
    for s in steps:
        x01_pert = perturb_topk(x01, heatmap_224, frac=float(s), mode=perturb_mode)

        # converte para PIL -> passa pelo transform do timm (garante mesma norm)
        pil_pert = Image.fromarray((x01_pert.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
        x_pert = to_tensor_for_model(pil_pert)

        logits_p = model(x_pert)
        conf_p = softmax_confidence(logits_p, target_idx)

        confs.append(conf_p)
        drops.append(base_conf - conf_p)

    aopc_mean = float(np.mean(drops)) if len(drops) else 0.0
    return {
        "pred_idx": pred_idx,
        "target_idx": target_idx,
        "base_conf": base_conf,
        "steps": list(steps),
        "confs": confs,
        "drops": drops,
        "aopc_mean": aopc_mean,
        "perturb_mode": perturb_mode,
        "target": target
    }

EPS = 1e-12

# VERIFICAR SE TEM MÉTRICA DIFERENTE DO VIT, SEN APENAS USA A MESMA 

def ensure_2d(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm)
    if hm.ndim == 3:
        if hm.shape[0] == 1:
            hm = hm[0]
        elif hm.shape[-1] == 1:
            hm = hm[...,0]
        else:
            hm = hm.mean(axis=-1)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap precisa ser 2D, recebi shape={hm.shape}")
    return hm.astype(np.float32)

def mass_norm(hm: np.ndarray) -> np.ndarray:
    hm = ensure_2d(hm)
    hm = np.maximum(hm, 0.0)
    s = hm.sum()
    if s <= 0:
        return np.full_like(hm, 1.0/hm.size, dtype=np.float32)
    return (hm / (s + EPS)).astype(np.float32)

def area_at_alpha(hm: np.ndarray, alpha: float) -> float:
    if not (0 < alpha <= 1):
        raise ValueError("alpha deve estar em (0,1].")
    mass = mass_norm(hm).reshape(-1)
    flat_sorted = np.sort(mass)[::-1]
    cumsum = np.cumsum(flat_sorted)
    k = int(np.searchsorted(cumsum, alpha) + 1)
    return float(k / mass.size)

def mpl_curve(hm: np.ndarray, area_grid=None):
    mass = mass_norm(hm).reshape(-1)
    flat_sorted = np.sort(mass)[::-1]
    cumsum = np.cumsum(flat_sorted)

    if area_grid is None:
        area_grid = np.linspace(0.01, 1.0, 50, dtype=np.float32)

    ks = np.clip((area_grid * mass.size).astype(int), 1, mass.size)
    mass_captured = np.array([cumsum[k-1] for k in ks], dtype=np.float32)
    return area_grid, mass_captured

def auc_trapz(x, y) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return float(np.trapezoid(y, x))

def mpl_auc(hm: np.ndarray, area_grid=None) -> float:
    x, y = mpl_curve(hm, area_grid=area_grid)
    return auc_trapz(x, y)

def entropy_map(hm: np.ndarray) -> float:
    mass = mass_norm(hm).reshape(-1)
    mass = mass[mass > 0]
    if mass.size == 0:
        return 0.0
    return float(-(mass * np.log(mass + EPS)).sum())

def compute_locality_metrics(hm: np.ndarray, do_entropy=True) -> dict:
    out = {
        "Area@50": area_at_alpha(hm, 0.50),
        "Area@90": area_at_alpha(hm, 0.90),
        "MPL_AUC": mpl_auc(hm),
    }
    if do_entropy:
        out["Entropy"] = entropy_map(hm)
    return out

# Escolha uma imagem do df (igual você fazia no ViT)
IDX = 0
img_path = df.loc[IDX, "file"]
true_lbl = int(df.loc[IDX, "label"])

pil = load_image(img_path)
x = to_tensor_for_model(pil)
with torch.no_grad():
    logits = model(x)
pred = int(logits.argmax(dim=1).item())
conf = softmax_confidence(logits, pred)

print("IMG:", img_path)
print("true:", true_lbl, "pred:", pred, "conf:", f"{conf:.4f}")

# gera mapas CNN-specific
maps = {
    "GradCAM":   cam_map_for_image(pil, target_class=pred, method="gradcam"),
    "GradCAM++": cam_map_for_image(pil, target_class=pred, method="gradcampp"),
    "LayerCAM":  cam_map_for_image(pil, target_class=pred, method="layercam"),
    "ScoreCAM":  scorecam_map_for_image(pil, target_class=pred),
}
{k: v.shape for k,v in maps.items()}