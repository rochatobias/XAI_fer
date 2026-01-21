# ==============================================================================
# Data Loader - Carregamento de dataset para XAI
# ==============================================================================

import os
import random
from typing import List, Dict, Optional
import pandas as pd
from PIL import Image

from config import DATA_DIR, EMOTION_CLASSES, N_SAMPLES


def get_all_images(data_dir: str = DATA_DIR) -> List[Dict]:
    """Carrega todos os caminhos de imagens do diretório de dados."""
    images = []
    for label_idx, label in enumerate(EMOTION_CLASSES):
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append({
                    'path': os.path.join(class_dir, filename),
                    'label': label,
                    'label_idx': label_idx,
                    'filename': filename
                })
    return images


def sample_images(images: List[Dict], n_samples: int = N_SAMPLES, seed: int = 42, balanced: bool = True) -> List[Dict]:
    """Amostra n_samples imagens do dataset."""
    random.seed(seed)
    if n_samples >= len(images):
        return images
    if balanced:
        by_class = {}
        for img in images:
            label = img['label']
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(img)
        n_classes = len(by_class)
        per_class = max(1, n_samples // n_classes)
        remainder = n_samples % n_classes
        sampled = []
        for i, (label, class_images) in enumerate(by_class.items()):
            n_from_class = per_class + (1 if i < remainder else 0)
            n_from_class = min(n_from_class, len(class_images))
            sampled.extend(random.sample(class_images, n_from_class))
        return sampled[:n_samples]
    else:
        return random.sample(images, n_samples)


def load_dataset(n_samples: Optional[int] = None, data_dir: str = DATA_DIR, balanced: bool = True, seed: int = 42) -> pd.DataFrame:
    """Função principal para carregar o dataset."""
    if n_samples is None:
        n_samples = N_SAMPLES
    print(f"Carregando imagens de: {data_dir}")
    all_images = get_all_images(data_dir)
    print(f"Total de imagens encontradas: {len(all_images)}")
    sampled = sample_images(all_images, n_samples, seed, balanced)
    print(f"Imagens amostradas: {len(sampled)}")
    df = pd.DataFrame(sampled)
    print("\nDistribuição por classe:")
    for label in EMOTION_CLASSES:
        count = len(df[df['label'] == label])
        if count > 0:
            print(f"  {label}: {count}")
    return df


def load_single_image(image_path: str) -> Image.Image:
    """Carrega uma única imagem do disco."""
    return Image.open(image_path).convert("RGB")
