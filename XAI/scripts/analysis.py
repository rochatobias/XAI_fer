# ==============================================================================
# Analysis - Geração de CSVs para Análise de Pesquisa
# ==============================================================================
# Filtra resultados por cenários de interesse (confiança alta/baixa, acertos/erros)
# Gera estatísticas por método XAI e por classe de emoção
# ==============================================================================

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config import (
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD,
    ANALYSIS_DIR, EMOTION_CLASSES
)


class XAIAnalyzer:
    """Classe para análise e geração de CSVs de pesquisa."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: str = ANALYSIS_DIR):
        """
        Inicializa o analisador.
        
        Args:
            results_df: DataFrame com resultados do pipeline XAI
            output_dir: Diretório para salvar CSVs
        """
        self.df = results_df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # Filtros de Cenários de Pesquisa
    # ==========================================================================
    
    def filter_high_conf_errors(self, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        """
        Filtra casos: alta confiança + errado.
        
        Útil para analisar quando o modelo "está certeza mas erra" - casos problemáticos.
        """
        return self.df[(~self.df["correct"]) & (self.df["conf"] >= threshold)].copy()
    
    def filter_low_conf_errors(self, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        """
        Filtra casos: baixa confiança + errado.
        
        Modelo não tinha certeza e realmente errou - pode indicar imagens difíceis.
        """
        return self.df[(~self.df["correct"]) & (self.df["conf"] < threshold)].copy()
    
    def filter_low_conf_correct(self, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        """
        Filtra casos: baixa confiança + correto.
        
        Modelo não tinha certeza mas acertou - pode indicar sorte ou features fracas.
        """
        return self.df[(self.df["correct"]) & (self.df["conf"] < threshold)].copy()
    
    def filter_high_conf_correct(self, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
        """
        Filtra casos: alta confiança + correto.
        
        Casos ideais - modelo confiante e correto.
        """
        return self.df[(self.df["correct"]) & (self.df["conf"] >= threshold)].copy()
    
    # ==========================================================================
    # Estatísticas
    # ==========================================================================
    
    def compute_stats_by_method(self) -> pd.DataFrame:
        """
        Calcula média e desvio padrão das métricas por método XAI.
        
        Returns:
            DataFrame com média e DP de cada métrica por método
        """
        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "MPL_AUC", "Entropy", "Gini"
        ]]
        
        if not metric_cols:
            # Fallback para nomes antigos
            metric_cols = [c for c in self.df.columns if c in [
                "AOPC", "Insertion_AUC", "Deletion_AUC",
                "Area@50", "Area@90", "MPL_AUC", "Entropy", "Gini"
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
        """
        Calcula média e desvio padrão das métricas por classe de emoção.
        
        Returns:
            DataFrame com média e DP de cada métrica por classe
        """
        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "MPL_AUC", "Entropy", "Gini", "conf"
        ]]
        
        stats = []
        for label in self.df["label"].unique():
            class_df = self.df[self.df["label"] == label]
            row = {"class": label}
            row["n_samples"] = len(class_df["image_idx"].unique())
            row["accuracy"] = class_df.groupby("image_idx")["correct"].first().mean()
            for col in metric_cols:
                if col in class_df.columns:
                    row[f"{col}_mean"] = class_df[col].mean()
                    row[f"{col}_std"] = class_df[col].std()
            stats.append(row)
        
        return pd.DataFrame(stats)
    
    def compute_stats_by_model(self) -> pd.DataFrame:
        """
        Calcula estatísticas por modelo (se houver coluna 'model').
        """
        if "model" not in self.df.columns:
            return pd.DataFrame()
        
        metric_cols = [c for c in self.df.columns if c in [
            "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
            "Area@50", "Area@90", "MPL_AUC", "Entropy", "Gini", "conf"
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
    
    # ==========================================================================
    # Geração de CSVs
    # ==========================================================================
    
    def save_analysis_csvs(self, verbose: bool = True) -> Dict[str, str]:
        """
        Gera todos os CSVs de análise.
        
        Returns:
            Dicionário {nome: caminho} dos CSVs gerados
        """
        generated = {}
        
        # 1. Casos filtrados por confiança/acerto
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
        
        # 2. Estatísticas por método
        stats_method = self.compute_stats_by_method()
        if len(stats_method) > 0:
            path = os.path.join(self.output_dir, "metrics_by_method.csv")
            stats_method.to_csv(path, index=False)
            generated["metrics_by_method"] = path
            if verbose:
                print(f"  metrics_by_method: {len(stats_method)} métodos -> {path}")
        
        # 3. Estatísticas por classe
        stats_class = self.compute_stats_by_class()
        if len(stats_class) > 0:
            path = os.path.join(self.output_dir, "metrics_by_class.csv")
            stats_class.to_csv(path, index=False)
            generated["metrics_by_class"] = path
            if verbose:
                print(f"  metrics_by_class: {len(stats_class)} classes -> {path}")
        
        # 4. Estatísticas por modelo (se aplicável)
        stats_model = self.compute_stats_by_model()
        if len(stats_model) > 0:
            path = os.path.join(self.output_dir, "metrics_by_model.csv")
            stats_model.to_csv(path, index=False)
            generated["metrics_by_model"] = path
            if verbose:
                print(f"  metrics_by_model: {len(stats_model)} modelos -> {path}")
        
        return generated


def generate_analysis_report(
    results_df: pd.DataFrame,
    output_dir: str = ANALYSIS_DIR,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Função conveniente para gerar todos os CSVs de análise.
    
    Args:
        results_df: DataFrame com resultados do pipeline
        output_dir: Diretório de saída
        verbose: Se True, imprime progresso
    
    Returns:
        Dicionário com caminhos dos arquivos gerados
    """
    if verbose:
        print("\n[ANÁLISE] Gerando CSVs de análise...")
    
    analyzer = XAIAnalyzer(results_df, output_dir)
    return analyzer.save_analysis_csvs(verbose=verbose)
