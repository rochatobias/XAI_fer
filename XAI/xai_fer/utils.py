"""
Utilitários comuns do pacote xai_fer.

Funções de mapeamento entre índices numéricos de classe e seus nomes textuais,
além de helpers para enriquecer DataFrames de resultados com labels e flags
de acerto/erro.
"""

import pandas as pd
from typing import Dict, Optional

from xai_fer.config import EMOTION_CLASSES


def get_label_name(label_idx: int) -> str:
    """Converte índice numérico para nome da classe (ex: 3 → 'happy')."""
    if 0 <= label_idx < len(EMOTION_CLASSES):
        return EMOTION_CLASSES[label_idx]
    return f"unknown_{label_idx}"


def get_label_idx(label_name: str) -> int:
    """Converte nome da classe para índice numérico (ex: 'happy' → 3)."""
    if label_name in EMOTION_CLASSES:
        return EMOTION_CLASSES.index(label_name)
    return -1


def enrich_df_with_labels(
    df: pd.DataFrame,
    idx2label: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Enriquece o DataFrame com colunas 'pred_label' e 'correct'.

    Útil para pós-processar CSVs de métricas que contêm apenas o índice
    numérico da predição (``pred_idx``).

    Args:
        df: DataFrame contendo ao menos a coluna ``pred_idx``.
        idx2label: Mapeamento personalizado {idx: nome}. Se ``None``,
                   usa ``EMOTION_CLASSES``.

    Returns:
        Cópia do DataFrame com as colunas extras adicionadas.
    """
    if idx2label is None:
        idx2label = {i: c for i, c in enumerate(EMOTION_CLASSES)}
    df = df.copy()
    if "pred_idx" in df.columns:
        df["pred_label"] = df["pred_idx"].map(idx2label)
    if "label" in df.columns and "pred_label" in df.columns:
        df["correct"] = df["label"] == df["pred_label"]
    return df


def select_cases(
    df: pd.DataFrame,
    kind: str = "errors_confident",
    n: int = 5,
    conf_thr: float = 0.9,
) -> pd.DataFrame:
    """Seleciona casos interessantes do DataFrame para inspeção rápida.

    Args:
        df: DataFrame de resultados com colunas ``correct`` e ``conf``.
        kind: Tipo de caso a selecionar:
              ``'errors_confident'`` — erros com alta confiança,
              ``'hits_confident'``   — acertos com alta confiança,
              ``'low_conf'``         — casos de menor confiança geral.
        n: Número de casos a retornar.
        conf_thr: Threshold de confiança para filtros ``confident``.

    Returns:
        DataFrame filtrado com até *n* linhas.
    """
    if kind == "errors_confident":
        sub = df[(~df["correct"]) & (df["conf"] >= conf_thr)].sort_values("conf", ascending=False)
    elif kind == "hits_confident":
        sub = df[(df["correct"]) & (df["conf"] >= conf_thr)].sort_values("conf", ascending=False)
    elif kind == "low_conf":
        sub = df.sort_values("conf", ascending=True)
    else:
        raise ValueError(f"kind inválido: {kind}")
    return sub.head(n).reset_index(drop=True)
