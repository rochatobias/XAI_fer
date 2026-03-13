"""
Seleção estratificada de imagens para geração de heatmaps.

Estratégia: 7 classes × 4 buckets (Confiança/Acerto) × N imagens.

Os 4 buckets capturam cenários de pesquisa importantes:
    - ``correct_high``: acertou com alta confiança (caso ideal).
    - ``correct_low``: acertou com baixa confiança (possível sorte).
    - ``wrong_high``: errou com alta confiança (caso problemático).
    - ``wrong_low``: errou com baixa confiança (imagem ambígua).

Isso gera ~140 heatmaps representativos (7 × 4 × 5) ao invés de milhares,
permitindo análise qualitativa viável e comparação direta entre métodos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from xai_fer.config import (
    HEATMAPS_PER_CELL,
    PERCENTILE_HIGH,
    PERCENTILE_LOW,
    EMOTION_CLASSES,
)


class StratifiedSelector:
    """Seleciona subconjunto de imagens para heatmaps via amostragem estratificada.

    Os thresholds de confiança são calculados dinamicamente a partir dos
    percentis globais (P20 e P80 por padrão), adaptando-se ao dataset.

    Args:
        df: DataFrame com resultados (colunas: ``conf``, ``correct``,
            ``label``, ``image_idx``).
        n_per_cell: Número de imagens por célula (classe × bucket).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_per_cell: int = HEATMAPS_PER_CELL,
    ):
        self.df = df.copy()
        self.n = n_per_cell

    def _calculate_thresholds(self) -> Tuple[float, float]:
        """Calcula thresholds baseados nos percentis globais de confiança."""
        if len(self.df) < 10:
            return 0.8, 0.2
        high_thr = np.percentile(self.df["conf"], PERCENTILE_HIGH)
        low_thr = np.percentile(self.df["conf"], PERCENTILE_LOW)
        return high_thr, low_thr

    def select_candidates(self) -> pd.DataFrame:
        """Realiza a seleção estratificada.

        Returns:
            DataFrame contendo apenas as imagens selecionadas, com
            coluna extra ``selection_reason`` descrevendo o bucket.
        """
        if self.df.empty:
            return pd.DataFrame()

        high_thr, low_thr = self._calculate_thresholds()
        print(
            f"\n[SELEÇÃO] Thresholds (Global): "
            f"High (P{PERCENTILE_HIGH})={high_thr:.4f}, "
            f"Low (P{PERCENTILE_LOW})={low_thr:.4f}"
        )

        df_unique = self.df.groupby("image_idx").first().reset_index()

        selected_indices: List[int] = []
        selection_reasons: Dict[int, str] = {}

        for label in EMOTION_CLASSES:
            class_df = df_unique[df_unique["label"] == label]

            buckets = {
                "correct_high": class_df[
                    (class_df["correct"]) & (class_df["conf"] >= high_thr)
                ]["image_idx"].tolist(),
                "correct_low": class_df[
                    (class_df["correct"]) & (class_df["conf"] <= low_thr)
                ]["image_idx"].tolist(),
                "wrong_high": class_df[
                    (~class_df["correct"]) & (class_df["conf"] >= high_thr)
                ]["image_idx"].tolist(),
                "wrong_low": class_df[
                    (~class_df["correct"]) & (class_df["conf"] <= low_thr)
                ]["image_idx"].tolist(),
            }

            for bucket_name, indices in buckets.items():
                if not indices:
                    continue

                # Ordena por confiança (maiores para high, menores para low)
                if "high" in bucket_name:
                    confs = class_df[class_df["image_idx"].isin(indices)].set_index("image_idx")["conf"]
                    indices_sorted = confs.sort_values(ascending=False).index.tolist()
                elif "low" in bucket_name:
                    confs = class_df[class_df["image_idx"].isin(indices)].set_index("image_idx")["conf"]
                    indices_sorted = confs.sort_values(ascending=True).index.tolist()
                else:
                    indices_sorted = indices

                chosen = indices_sorted[:self.n]
                for idx in chosen:
                    if idx not in selection_reasons:
                        selected_indices.append(idx)
                        selection_reasons[idx] = f"{label}_{bucket_name}"

        final_selection = self.df[self.df["image_idx"].isin(selected_indices)].copy()
        final_selection["selection_reason"] = final_selection["image_idx"].map(selection_reasons)

        n_unique = len(selected_indices)
        print(f"[SELEÇÃO] Selecionadas {n_unique} imagens únicas para heatmaps.")

        return final_selection
