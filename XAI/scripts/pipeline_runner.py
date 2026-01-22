"""Pipeline Runner - Execução genérica de pipelines XAI.

Abstrai loop, execução XAI, heatmaps e métricas para evitar duplicação de código.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from PIL import Image

from config import (
    HEATMAPS_DIR, RESULTS_DIR, MEAN, STD,
    VIT_XAI_METHODS, CNN_XAI_METHODS, AGNOSTIC_XAI_METHODS
)
from utils import get_label_name
from metrics import compute_all_metrics
from visualization import save_xai_visualization

class XAIPipelineRunner:
    """Class genérica para rodar experimentos XAI em um dataset."""
    
    def __init__(
        self,
        model: Any,
        transform: Any,
        device: Any,
        model_type: str,
        xai_methods: Tuple[str, ...],
        xai_function: Any # Função que recebe (img_path, model, transform, device, methods)
    ):
        self.model = model
        self.transform = transform
        self.device = device
        self.model_type = model_type.lower()
        self.xai_methods = xai_methods
        self.xai_function = xai_function
        
        # Cria diretório de saída para heatmaps deste modelo
        self.heatmaps_dir = os.path.join(HEATMAPS_DIR, self.model_type)
        os.makedirs(self.heatmaps_dir, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        save_heatmaps: bool = True,
        save_metrics: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Executa o pipeline para todas as imagens no DataFrame.
        """
        all_results = []
        total = len(df)
        
        for idx, row in df.iterrows():
            img_path = row['path']
            true_label = row['label']
            true_label_idx = row['label_idx']
            filename = row['filename']
            # Usa índice original se existir (para manter consistência em sub-seleções), senão usa o loop idx
            image_idx = row.get('image_idx', idx) 
            
            if verbose:
                print(f"\n  [{idx+1}/{total}] {filename} (Label: {true_label})")
            
            try:
                # Executa XAI
                pil_img, pred_idx, conf, maps = self.xai_function(
                    img_path, self.model, self.transform, self.device, methods=self.xai_methods
                )
                pred_label = get_label_name(pred_idx)
                correct = (pred_idx == true_label_idx)
                
                if verbose:
                    status = "✓" if correct else "✗"
                    print(f"       Pred: {pred_label} ({conf:.2%}) {status}")
                
                # Salvar visualização
                if save_heatmaps:
                    status_str = "OK" if correct else "ERR"
                    heatmap_filename = f"{image_idx:03d}_{status_str}_{true_label}_pred{pred_label}_{conf:.2f}.png"
                    heatmap_path = os.path.join(self.heatmaps_dir, heatmap_filename)
                    title = f"{self.model_type.upper()} | {status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                    save_xai_visualization(pil_img, maps, heatmap_path, title, show=False)
                
                # Calcular métricas (apenas se solicitado)
                if save_metrics:
                    for method_name, heatmap in maps.items():
                        if verbose:
                            print(f"       Métricas: {method_name}...")
                        metrics = compute_all_metrics(
                            self.model, pil_img, heatmap, self.device, MEAN, STD,
                            model_type=self.model_type, true_label_idx=true_label_idx
                        )
                        result = {
                            'model': self.model_type.upper(), # 'VIT' ou 'CNN'
                            'image_idx': image_idx,
                            'filename': filename,
                            'path': img_path,
                            'label': true_label,
                            'label_idx': true_label_idx,
                            'pred_idx': pred_idx,
                            'pred_label': pred_label,
                            'conf': conf,
                            'correct': correct,
                            'method': method_name,
                            **metrics
                        }
                        all_results.append(result)
                    
            except Exception as e:
                print(f"       ERRO: {e}")
                import traceback
                traceback.print_exc()
        
        results_df = pd.DataFrame(all_results)
        
        if save_metrics and not results_df.empty:
            csv_path = os.path.join(RESULTS_DIR, f"metrics_{self.model_type}.csv")
            results_df.to_csv(csv_path, index=False)
            if verbose:
                print(f"\n[{self.model_type.upper()}] Métricas salvas em: {csv_path}")
                
        return results_df
