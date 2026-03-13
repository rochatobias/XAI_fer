"""
Pipeline Runner — orquestração genérica de experimentos XAI.

Abstrai o loop de processamento: para cada imagem do DataFrame, executa
a função XAI do modelo, opcionalmente salva heatmaps e calcula métricas
de fidelidade/localidade.

Projetado para ser reutilizável com qualquer modelo (ViT, CNN), bastando
fornecer a ``xai_function`` adequada na inicialização.
"""

import os
import traceback
from typing import Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from xai_fer.config import HEATMAPS_DIR, RESULTS_DIR, MEAN, STD
from xai_fer.utils import get_label_name
from xai_fer.evaluation.metrics import compute_all_metrics
from xai_fer.visualization.heatmaps import save_xai_visualization


class XAIPipelineRunner:
    """Classe genérica para rodar experimentos XAI em um dataset.

    Recebe modelo, transformações e função XAI e processa todas as
    imagens de um DataFrame, gerando métricas e/ou heatmaps.

    Args:
        model: Modelo PyTorch (ViT ou CNN).
        transform: Pipeline de pré-processamento.
        device: Device do modelo.
        model_type: ``'vit'`` ou ``'cnn'``.
        xai_methods: Tupla de nomes dos métodos a executar.
        xai_function: Função com assinatura
            ``(img_path, model, transform, device, methods) → (pil_img, pred, conf, maps)``.
    """

    def __init__(
        self,
        model: Any,
        transform: Any,
        device: Any,
        model_type: str,
        xai_methods: Tuple[str, ...],
        xai_function: Any,
    ):
        self.model = model
        self.transform = transform
        self.device = device
        self.model_type = model_type.lower()
        self.xai_methods = xai_methods
        self.xai_function = xai_function

        self.heatmaps_dir = os.path.join(HEATMAPS_DIR, self.model_type)
        os.makedirs(self.heatmaps_dir, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        save_heatmaps: bool = True,
        save_metrics: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Executa o pipeline para todas as imagens no DataFrame.

        Args:
            df: DataFrame com colunas ``path``, ``label``, ``label_idx``,
                ``filename``.
            save_heatmaps: Se ``True``, salva visualizações de heatmaps.
            save_metrics: Se ``True``, calcula e salva métricas em CSV.
            verbose: Se ``True``, imprime progresso por imagem.

        Returns:
            DataFrame com métricas por ``(imagem, método)``.
        """
        all_results = []
        total = len(df)

        for idx, row in df.iterrows():
            img_path = row["path"]
            true_label = row["label"]
            true_label_idx = row["label_idx"]
            filename = row["filename"]
            image_idx = row.get("image_idx", idx)

            if verbose:
                print(f"\n  [{idx + 1}/{total}] {filename} (Label: {true_label})")

            try:
                pil_img, pred_idx, conf, maps = self.xai_function(
                    img_path, self.model, self.transform, self.device,
                    methods=self.xai_methods,
                )
                pred_label = get_label_name(pred_idx)
                correct = pred_idx == true_label_idx

                if verbose:
                    status = "✓" if correct else "✗"
                    print(f"       Pred: {pred_label} ({conf:.2%}) {status}")

                if save_heatmaps:
                    status_str = "OK" if correct else "ERR"
                    heatmap_filename = (
                        f"{image_idx:03d}_{status_str}_{true_label}"
                        f"_pred{pred_label}_{conf:.2f}.png"
                    )
                    heatmap_path = os.path.join(self.heatmaps_dir, heatmap_filename)
                    title = (
                        f"{self.model_type.upper()} | {status_str} | "
                        f"True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                    )
                    save_xai_visualization(pil_img, maps, heatmap_path, title, show=False)

                if save_metrics:
                    for method_name, heatmap in maps.items():
                        if verbose:
                            print(f"       Métricas: {method_name}...")
                        metrics = compute_all_metrics(
                            self.model, pil_img, heatmap, self.device, MEAN, STD,
                            model_type=self.model_type,
                            true_label_idx=true_label_idx,
                        )
                        result = {
                            "model": self.model_type.upper(),
                            "image_idx": image_idx,
                            "filename": filename,
                            "path": img_path,
                            "label": true_label,
                            "label_idx": true_label_idx,
                            "pred_idx": pred_idx,
                            "pred_label": pred_label,
                            "conf": conf,
                            "correct": correct,
                            "method": method_name,
                            **metrics,
                        }
                        all_results.append(result)

            except Exception as e:
                print(f"       ERRO: {e}")
                traceback.print_exc()

        results_df = pd.DataFrame(all_results)

        if save_metrics and not results_df.empty:
            csv_path = os.path.join(RESULTS_DIR, f"metrics_{self.model_type}.csv")
            results_df.to_csv(csv_path, index=False)
            if verbose:
                print(f"\n[{self.model_type.upper()}] Métricas salvas em: {csv_path}")

        return results_df
