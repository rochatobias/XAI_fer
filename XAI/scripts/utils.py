"""Utils - Funções Utilitárias."""

import pandas as pd
from typing import Dict, Optional
from config import EMOTION_CLASSES


def enrich_df_with_labels(df: pd.DataFrame, idx2label: Optional[Dict] = None) -> pd.DataFrame:
    """Enriquece o DataFrame com labels de predição e flag de acerto."""
    if idx2label is None:
        idx2label = {i: c for i, c in enumerate(EMOTION_CLASSES)}
    df = df.copy()
    if 'pred_idx' in df.columns:
        df["pred_label"] = df["pred_idx"].map(idx2label)
    if 'label' in df.columns and 'pred_label' in df.columns:
        df["correct"] = df["label"] == df["pred_label"]
    return df


def select_cases(df: pd.DataFrame, kind: str = "errors_confident", n: int = 5, conf_thr: float = 0.9) -> pd.DataFrame:
    """Seleciona casos interessantes do DataFrame."""
    if kind == "errors_confident":
        sub = df[(~df["correct"]) & (df["conf"] >= conf_thr)].sort_values("conf", ascending=False)
    elif kind == "hits_confident":
        sub = df[(df["correct"]) & (df["conf"] >= conf_thr)].sort_values("conf", ascending=False)
    elif kind == "low_conf":
        sub = df.sort_values("conf", ascending=True)
    else:
        raise ValueError(f"kind inválido: {kind}")
    return sub.head(n).reset_index(drop=True)


def get_label_name(label_idx: int) -> str:
    """Retorna o nome da classe dado o índice."""
    if 0 <= label_idx < len(EMOTION_CLASSES):
        return EMOTION_CLASSES[label_idx]
    return f"unknown_{label_idx}"


def get_label_idx(label_name: str) -> int:
    """Retorna o índice dado o nome da classe."""
    if label_name in EMOTION_CLASSES:
        return EMOTION_CLASSES.index(label_name)
    return -1
