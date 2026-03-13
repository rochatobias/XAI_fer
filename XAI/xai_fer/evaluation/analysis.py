"""
Análise estatística e geração de CSVs para a pesquisa.

Filtra resultados por cenários de interesse (alta/baixa confiança × acerto/erro)
e gera estatísticas descritivas agrupadas por método XAI, classe de emoção e
modelo. Inclui bootstrap para intervalos de confiança de 95 % e detecção de
discrepâncias entre ViT e CNN.

Os CSVs gerados são insumos diretos para tabelas do paper.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from xai_fer.config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    ANALYSIS_DIR,
    EMOTION_CLASSES,
)


class XAIAnalyzer:
    """Classe para análise estatística e geração de CSVs de pesquisa.

    Recebe o DataFrame consolidado de métricas (``metrics_combined.csv``)
    e oferece filtros por cenário e cálculos de estatísticas descritivas.

    Args:
        results_df: DataFrame com resultados do pipeline XAI.
        output_dir: Diretório para salvar CSVs gerados.
    """

    def __init__(
        self,
        results_df: pd.DataFrame,
        output_dir: str = ANALYSIS_DIR,
    ):
        self.df = results_df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ==========================================================================
    # Filtros de Cenários de Pesquisa
    # ==========================================================================

    def filter_high_conf_errors(
        self, threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Alta confiança + errado — casos problemáticos onde o modelo 'tem
        certeza mas erra'. Indicam viés ou confusão sistemática."""
        return self.df[(~self.df["correct"]) & (self.df["conf"] >= threshold)].copy()

    def filter_low_conf_errors(
        self, threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Baixa confiança + errado — imagens difíceis ou ambíguas."""
        return self.df[(~self.df["correct"]) & (self.df["conf"] < threshold)].copy()

    def filter_low_conf_correct(
        self, threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Baixa confiança + correto — possível 'sorte' ou features fracas."""
        return self.df[(self.df["correct"]) & (self.df["conf"] < threshold)].copy()

    def filter_high_conf_correct(
        self, threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Alta confiança + correto — caso ideal de funcionamento."""
        return self.df[(self.df["correct"]) & (self.df["conf"] >= threshold)].copy()

    # ==========================================================================
    # Estatísticas
    # ==========================================================================

    def compute_stats_by_method(self) -> pd.DataFrame:
        """Média e desvio padrão de cada métrica, agrupados por método XAI."""
        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "Concentration_AUC", "Entropy", "Gini",
        ]]
        if not metric_cols:
            metric_cols = [c for c in self.df.columns if c in [
                "AOPC", "Insertion_AUC", "Deletion_AUC",
                "Area@50", "Area@90", "Concentration_AUC", "Entropy", "Gini",
            ]]

        stats = []
        for method in self.df["method"].unique():
            method_df = self.df[self.df["method"] == method]
            row = {"method": method}
            for col in metric_cols:
                if col in method_df.columns:
                    row[f"{col}_mean"] = method_df[col].mean()
                    row[f"{col}_std"] = method_df[col].std()
            stats.append(row)
        return pd.DataFrame(stats)

    def compute_stats_by_class(self) -> pd.DataFrame:
        """Média e desvio padrão por classe de emoção, separado por modelo."""
        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "Concentration_AUC", "Entropy", "Gini", "conf",
        ]]
        stats = []
        models = self.df["model"].unique() if "model" in self.df.columns else ["ALL"]

        for model in models:
            model_df = self.df[self.df["model"] == model] if model != "ALL" else self.df
            for label in model_df["label"].unique():
                class_df = model_df[model_df["label"] == label]
                row = {"model": model, "class": label}
                row["n_samples"] = len(class_df["image_idx"].unique())
                row["accuracy"] = class_df.groupby("image_idx")["correct"].first().mean()
                for col in metric_cols:
                    if col in class_df.columns:
                        row[f"{col}_mean"] = class_df[col].mean()
                        row[f"{col}_std"] = class_df[col].std()
                stats.append(row)
        return pd.DataFrame(stats)

    def compute_stats_by_model(self) -> pd.DataFrame:
        """Estatísticas por modelo (ViT vs CNN)."""
        if "model" not in self.df.columns:
            return pd.DataFrame()

        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "Concentration_AUC", "Entropy", "Gini", "conf",
        ]]
        stats = []
        for model_name in self.df["model"].unique():
            model_df = self.df[self.df["model"] == model_name]
            row = {"model": model_name}
            row["n_samples"] = len(model_df["image_idx"].unique())
            row["accuracy"] = model_df.groupby("image_idx")["correct"].first().mean()
            for col in metric_cols:
                if col in model_df.columns:
                    row[f"{col}_mean"] = model_df[col].mean()
                    row[f"{col}_std"] = model_df[col].std()
            stats.append(row)
        return pd.DataFrame(stats)

    def compute_model_discrepancies(self) -> pd.DataFrame:
        """Identifica imagens onde ViT e CNN discordam na predição.

        Muito útil para pesquisa: comparar XAI onde modelos divergem
        pode revelar diferenças nas features aprendidas.
        """
        if "model" not in self.df.columns:
            return pd.DataFrame()

        pivot = self.df.groupby(["image_idx", "model"]).agg({
            "filename": "first",
            "label": "first",
            "pred_label": "first",
            "pred_idx": "first",
            "conf": "first",
            "correct": "first",
        }).reset_index()

        discrepancies = []
        for img_idx in pivot["image_idx"].unique():
            img_data = pivot[pivot["image_idx"] == img_idx]
            models_data = {}
            for _, row in img_data.iterrows():
                models_data[row["model"]] = row

            if len(models_data) >= 2:
                model_names = list(models_data.keys())
                pred_labels = [models_data[m]["pred_label"] for m in model_names]
                if len(set(pred_labels)) > 1:
                    first = models_data[model_names[0]]
                    disc = {
                        "image_idx": img_idx,
                        "filename": first["filename"],
                        "true_label": first["label"],
                    }
                    for m in model_names:
                        disc[f"{m}_pred"] = models_data[m]["pred_label"]
                        disc[f"{m}_conf"] = models_data[m]["conf"]
                        disc[f"{m}_correct"] = models_data[m]["correct"]
                    discrepancies.append(disc)

        return pd.DataFrame(discrepancies)

    def compute_bootstrap_stats(
        self,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """Calcula Bootstrap Confidence Interval (95 %) para métricas.

        Agrupa por (Modelo, Método) e gera intervalos de confiança via
        reamostragem, útil para verificar significância estatística no paper.
        """
        available_cols = self.df.columns.tolist()
        target_metrics = [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "MPL_AUC", "Entropy",
        ]
        metric_cols = [c for c in target_metrics if c in available_cols]
        if not metric_cols:
            return pd.DataFrame()

        stats = []
        alpha = 1.0 - ci
        lower_p = alpha / 2.0 * 100
        upper_p = (1.0 - alpha / 2.0) * 100

        groups = self.df.groupby(["model", "method"])
        for (model, method), group_df in groups:
            n_samples = len(group_df)
            if n_samples < 5:
                continue

            row_base = {"model": model, "method": method, "n_samples": n_samples}
            data_np = {col: group_df[col].to_numpy() for col in metric_cols}

            boot_means = {col: [] for col in metric_cols}
            rng = np.random.default_rng(42)

            for _ in range(n_bootstraps):
                indices = rng.integers(0, n_samples, n_samples)
                for col in metric_cols:
                    boot_means[col].append(np.mean(data_np[col][indices]))

            for col in metric_cols:
                means = np.array(boot_means[col])
                out_name = "Concentration_AUC" if col == "MPL_AUC" else col
                row_base[f"{out_name}_mean"] = np.mean(data_np[col])
                row_base[f"{out_name}_ci_lower"] = np.percentile(means, lower_p)
                row_base[f"{out_name}_ci_upper"] = np.percentile(means, upper_p)

            stats.append(row_base)
        return pd.DataFrame(stats)

    # ==========================================================================
    # Geração de CSVs
    # ==========================================================================

    def save_analysis_csvs(self, verbose: bool = True) -> Dict[str, str]:
        """Gera e salva todos os CSVs de análise no diretório de saída.

        Returns:
            Dicionário ``{nome: caminho}`` dos CSVs gerados.
        """
        generated = {}

        filters = [
            ("high_conf_errors", self.filter_high_conf_errors()),
            ("low_conf_errors", self.filter_low_conf_errors()),
            ("low_conf_correct", self.filter_low_conf_correct()),
            ("high_conf_correct", self.filter_high_conf_correct()),
        ]

        for name, df in filters:
            if len(df) > 0:
                path = os.path.join(self.output_dir, f"{name}.csv")
                df.to_csv(path, index=False)
                generated[name] = path
                if verbose:
                    print(f"  {name}: {len(df)} registros -> {path}")
            elif verbose:
                print(f"  {name}: 0 registros (não gerado)")

        stats_method = self.compute_stats_by_method()
        if len(stats_method) > 0:
            path = os.path.join(self.output_dir, "metrics_by_method.csv")
            stats_method.to_csv(path, index=False)
            generated["metrics_by_method"] = path
            if verbose:
                print(f"  metrics_by_method: {len(stats_method)} métodos -> {path}")

        stats_class = self.compute_stats_by_class()
        if len(stats_class) > 0:
            path = os.path.join(self.output_dir, "metrics_by_class.csv")
            stats_class.to_csv(path, index=False)
            generated["metrics_by_class"] = path
            if verbose:
                print(f"  metrics_by_class: {len(stats_class)} classes -> {path}")

        stats_model = self.compute_stats_by_model()
        if len(stats_model) > 0:
            path = os.path.join(self.output_dir, "metrics_by_model.csv")
            stats_model.to_csv(path, index=False)
            generated["metrics_by_model"] = path
            if verbose:
                print(f"  metrics_by_model: {len(stats_model)} modelos -> {path}")

        if verbose:
            print("  Calculando Bootstrap IC 95%...")
        stats_boot = self.compute_bootstrap_stats()
        if len(stats_boot) > 0:
            path = os.path.join(self.output_dir, "metrics_bootstrap_ci95.csv")
            stats_boot.to_csv(path, index=False)
            generated["metrics_bootstrap_ci95"] = path
            if verbose:
                print(f"  metrics_bootstrap_ci95: {len(stats_boot)} configurações -> {path}")

        discrepancies = self.compute_model_discrepancies()
        if len(discrepancies) > 0:
            path = os.path.join(self.output_dir, "model_discrepancies.csv")
            discrepancies.to_csv(path, index=False)
            generated["model_discrepancies"] = path
            if verbose:
                print(f"  model_discrepancies: {len(discrepancies)} imagens -> {path}")
        elif verbose:
            print("  model_discrepancies: 0 discrepâncias (modelos concordam)")

        return generated


def generate_analysis_report(
    results_df: pd.DataFrame,
    output_dir: str = ANALYSIS_DIR,
    verbose: bool = True,
) -> Dict[str, str]:
    """Função conveniente para gerar todos os CSVs de análise.

    Args:
        results_df: DataFrame com resultados do pipeline.
        output_dir: Diretório de saída.
        verbose: Se ``True``, imprime progresso.

    Returns:
        Dicionário com caminhos dos arquivos gerados.
    """
    if verbose:
        print("\n[ANÁLISE] Gerando CSVs de análise...")
    analyzer = XAIAnalyzer(results_df, output_dir)
    return analyzer.save_analysis_csvs(verbose=verbose)
