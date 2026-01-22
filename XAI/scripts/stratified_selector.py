# ==============================================================================
# Stratified Selector - Seleção Inteligente de Heatmaps
# ==============================================================================
# Seleciona imagens representativas para geração de heatmaps
# Estratégia: 7 classes x 4 buckets (Confiança/Acerto) x N imagens
# ==============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from config import (
    HEATMAPS_PER_CELL,
    PERCENTILE_HIGH,
    PERCENTILE_LOW,
    EMOTION_CLASSES
)

class StratifiedSelector:
    """
    Seleciona um subconjunto de imagens para geração de heatmaps
    baseado em critérios estratificados de confiança e acerto.
    """
    
    def __init__(self, df: pd.DataFrame, n_per_cell: int = HEATMAPS_PER_CELL):
        """
        Args:
            df: DataFrame com resultados (deve conter 'conf', 'correct', 'label', 'image_idx')
            n_per_cell: Número de imagens por célula (combinação classe/bucket)
        """
        self.df = df.copy()
        self.n = n_per_cell
        self.buckets = {
            "correct_high": [],
            "correct_low": [],
            "wrong_high": [],
            "wrong_low": []
        }
        
    def _calculate_thresholds(self) -> Tuple[float, float]:
        """Calcula thresholds baseados nos percentis globais de confiança."""
        # Se tivermos poucos dados, usamos thresholds fixos como fallback
        if len(self.df) < 10:
            return 0.8, 0.2
            
        high_thr = np.percentile(self.df["conf"], PERCENTILE_HIGH)
        low_thr = np.percentile(self.df["conf"], PERCENTILE_LOW)
        return high_thr, low_thr

    def select_candidates(self) -> pd.DataFrame:
        """
        Realiza a seleção estratificada.
        
        Returns:
            DataFrame contendo apenas as imagens selecionadas, com coluna extra 'selection_reason'
        """
        if self.df.empty:
            return pd.DataFrame()
            
        high_thr, low_thr = self._calculate_thresholds()
        print(f"\n[SELEÇÃO] Thresholds calculados (Global): High (P{PERCENTILE_HIGH})={high_thr:.4f}, Low (P{PERCENTILE_LOW})={low_thr:.4f}")
        
        # Garante unicidade por imagem (usa a primeira ocorrência se houver duplicatas por método)
        df_unique = self.df.groupby("image_idx").first().reset_index()
        
        selected_indices = []
        selection_reasons = {}
        
        # Itera por classe para garantir representatividade
        for label in EMOTION_CLASSES:
            class_df = df_unique[df_unique["label"] == label]
            
            # Define os 4 buckets para esta classe
            buckets_indices = {
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
                ]["image_idx"].tolist()
            }
            
            # Seleciona N de cada bucket
            for bucket_name, indices in buckets_indices.items():
                if not indices:
                    continue
                    
                # Prioriza extremos? (mais confiantes dos high, menos confiantes dos low)
                # Por simplicidade e diversidade, vamos pegar aleatório ou os primeiros (já que são embaralhados na carga)
                # Mas para ser determinístico, vamos ordenar pela confiança
                
                indices_sorted = indices # Default
                
                # Se for High Confidence, pega os MAIORES
                if "high" in bucket_name:
                    # Precisamos recuperar as confianças para ordenar
                    confs = class_df[class_df["image_idx"].isin(indices)].set_index("image_idx")["conf"]
                    indices_sorted = confs.sort_values(ascending=False).index.tolist()
                
                # Se for Low Confidence, pega os MENORES
                elif "low" in bucket_name:
                    confs = class_df[class_df["image_idx"].isin(indices)].set_index("image_idx")["conf"]
                    indices_sorted = confs.sort_values(ascending=True).index.tolist()
                
                chosen = indices_sorted[:self.n]
                
                for idx in chosen:
                    if idx not in selection_reasons: # Evita duplicatas se lógica mudar
                        selected_indices.append(idx)
                        selection_reasons[idx] = f"{label}_{bucket_name}"
        
        # Filtra o DF original para manter apenas as imagens selecionadas
        # Importante: O DF original pode ter múltiplas linhas per imagem (uma por método XAI)
        # Queremos MANTER todas as linhas dessas imagens selecionadas
        final_selection = self.df[self.df["image_idx"].isin(selected_indices)].copy()
        
        # Adiciona razão da seleção
        final_selection["selection_reason"] = final_selection["image_idx"].map(selection_reasons)
        
        n_unique = len(selected_indices)
        print(f"[SELEÇÃO] Selecionadas {n_unique} imagens únicas para heatmaps.")
        
        return final_selection

    def save_selection_map(self, output_path: str):
        """Salva um CSV explicando por que cada imagem foi selecionada."""
        # TODO: Implementar se necessário, por enquanto o dataframe retornado já tem 'selection_reason'
        pass
