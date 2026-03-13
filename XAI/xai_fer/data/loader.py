"""
Carregamento e amostragem de dados para o pipeline XAI-FER.

Responsável por varrer o diretório de imagens (organizado em subpastas por
classe de emoção), construir um DataFrame com metadados de cada imagem e
oferecer amostragem balanceada ou filtrada.

Convenção esperada do dataset::

    data/aplicaçãoXAI/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
"""

import os
import random
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from xai_fer.config import DATA_DIR, EMOTION_CLASSES, N_SAMPLES


def get_all_images(data_dir: str = DATA_DIR) -> List[Dict]:
    """Varre o diretório de dados e retorna lista de dicionários com metadados.

    Cada dicionário contém:
        - ``path``: caminho absoluto da imagem
        - ``label``: nome textual da classe (pasta)
        - ``label_idx``: índice numérico da classe
        - ``filename``: nome do arquivo

    Args:
        data_dir: Raiz do dataset (contém subpastas por classe).

    Returns:
        Lista de dicionários, um por imagem encontrada.
    """
    images: List[Dict] = []
    for label_idx, label in enumerate(EMOTION_CLASSES):
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(
                    {
                        "path": os.path.join(class_dir, filename),
                        "label": label,
                        "label_idx": label_idx,
                        "filename": filename,
                    }
                )
    return images


def sample_images(
    images: List[Dict],
    n_samples: int = N_SAMPLES,
    seed: int = 42,
    balanced: bool = True,
) -> List[Dict]:
    """Amostra *n_samples* imagens do dataset, opcionalmente balanceando por classe.

    Quando ``balanced=True``, distribui igualmente entre as classes disponíveis,
    atribuindo eventuais sobras por round-robin.

    Args:
        images: Lista completa de metadados (saída de ``get_all_images``).
        n_samples: Total de imagens desejado.
        seed: Semente para reprodutibilidade.
        balanced: Se ``True``, amostra igualmente por classe.

    Returns:
        Sublista amostrada de metadados.
    """
    random.seed(seed)
    if n_samples >= len(images):
        return images

    if balanced:
        by_class: Dict[str, List[Dict]] = {}
        for img in images:
            by_class.setdefault(img["label"], []).append(img)

        n_classes = len(by_class)
        per_class = max(1, n_samples // n_classes)
        remainder = n_samples % n_classes

        sampled: List[Dict] = []
        for i, (_, class_images) in enumerate(by_class.items()):
            n_from_class = per_class + (1 if i < remainder else 0)
            n_from_class = min(n_from_class, len(class_images))
            sampled.extend(random.sample(class_images, n_from_class))
        return sampled[:n_samples]

    return random.sample(images, n_samples)


def load_dataset(
    n_samples: Optional[int] = None,
    data_dir: str = DATA_DIR,
    balanced: bool = True,
    seed: int = 42,
    target_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Função principal para carregar o dataset como DataFrame.

    Dois modos de operação:

    1. **Amostragem** (padrão): carrega todas as imagens e amostra
       ``n_samples`` de forma balanceada.
    2. **Filtragem**: se ``target_paths`` for fornecido, carrega apenas
       as imagens cujo caminho está na lista — útil para o Pass 2
       (seleção estratificada).

    Args:
        n_samples: Número de amostras (ignorado se ``target_paths`` fornecido).
        data_dir: Diretório raiz do dataset.
        balanced: Se ``True``, amostra balanceada entre classes.
        seed: Semente para reprodutibilidade.
        target_paths: Lista de caminhos absolutos para filtrar imagens específicas.

    Returns:
        DataFrame com colunas ``path``, ``label``, ``label_idx``, ``filename``.
    """
    print(f"Carregando imagens de: {data_dir}")
    all_images = get_all_images(data_dir)
    print(f"Total de imagens encontradas: {len(all_images)}")

    if target_paths is not None:
        target_set = set(target_paths)
        sampled = [img for img in all_images if img["path"] in target_set]
        print(f"Imagens filtradas por lista de alvo: {len(sampled)}")
    else:
        if n_samples is None:
            n_samples = N_SAMPLES
        sampled = sample_images(all_images, n_samples, seed, balanced)
        print(f"Imagens amostradas: {len(sampled)}")

    df = pd.DataFrame(sampled)
    print("\nDistribuição por classe:")
    for label in EMOTION_CLASSES:
        count = len(df[df["label"] == label])
        if count > 0:
            print(f"  {label}: {count}")
    return df


def load_single_image(image_path: str) -> Image.Image:
    """Carrega uma única imagem do disco em modo RGB."""
    return Image.open(image_path).convert("RGB")
