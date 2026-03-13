"""
Geração de tabelas LaTeX para inclusão no paper.

Gera tabela combinada de métricas de fidelidade e localidade, agrupadas
por regime de confiança × acerto/erro. Formato pronto para compilação
LaTeX com booktabs.
"""

import os

import pandas as pd
from xai_fer.config import RESULTS_DIR


def generate_latex_table() -> None:
    """Gera tabela LaTeX a partir do CSV consolidado de métricas.

    Lê ``metrics_combined.csv`` e imprime no console uma tabela formatada
    com 4 regimes (High-conf Correct/Error, Low-conf Correct/Error) ×
    6 métricas (Insertion, Deletion, AOPC, Concentration, Entropy, Area@50).

    Setas (↑/↓) indicam a direção desejada de cada métrica.
    """
    csv_path = os.path.join(RESULTS_DIR, "metrics_combined.csv")
    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "MPL_AUC" in df.columns and "Concentration_AUC" not in df.columns:
        df["Concentration_AUC"] = df["MPL_AUC"]

    print("Usando TODAS as imagens (dataset completo)...")

    HIGH_CONF_THRESH = 0.70
    LOW_CONF_THRESH = 0.40

    mask_hc_error   = (df["conf"] > HIGH_CONF_THRESH) & (~df["correct"])
    mask_lc_error   = (df["conf"] < LOW_CONF_THRESH)  & (~df["correct"])
    mask_lc_correct = (df["conf"] < LOW_CONF_THRESH)  & (df["correct"])
    mask_hc_correct = (df["conf"] > HIGH_CONF_THRESH) & (df["correct"])

    print(f"High-Conf (>0.7) Correct: {mask_hc_correct.sum()}")
    print(f"High-Conf (>0.7) Error:   {mask_hc_error.sum()}")
    print(f"Low-Conf (<0.4) Correct:  {mask_lc_correct.sum()}")
    print(f"Low-Conf (<0.4) Error:    {mask_lc_error.sum()}")

    regimes = {
        "High-conf (>0.7) Correct": df[mask_hc_correct],
        "High-conf (>0.7) Error":   df[mask_hc_error],
        "Low-conf (<0.4) Correct":  df[mask_lc_correct],
        "Low-conf (<0.4) Error":    df[mask_lc_error],
    }

    metrics = [
        "Insertion_AUC", "Deletion_AUC", "AOPC_mean",
        "Concentration_AUC", "Entropy", "Area@50",
    ]

    print("\n--- LaTeX TABLE ---\n")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Métricas de Fidelidade e Localidade agregadas por regime "
          r"de confiança e acerto. Média $\pm$ desvio padrão.}")
    print(r"\label{tab:error_confidence_combined}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{Fidelity Metrics}} "
          r"& \multicolumn{3}{c}{\textbf{Locality Metrics}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"\textbf{Regime} & \textbf{Insertion} $\uparrow$ "
          r"& \textbf{Deletion} $\downarrow$ & \textbf{AOPC} $\uparrow$ "
          r"& \textbf{Concentration} $\uparrow$ & \textbf{Entropy} $\downarrow$ "
          r"& \textbf{Area@50} $\downarrow$ \\")
    print(r"\midrule")

    for label, sub_df in regimes.items():
        row_str = f"{label}"
        for m in metrics:
            if m in sub_df.columns:
                valid_vals = sub_df[sub_df[m] > 1e-6][m] if m == "Entropy" else sub_df[m]
                if not valid_vals.empty:
                    mean_val = valid_vals.mean()
                    std_val = valid_vals.std()
                    row_str += f" & ${mean_val:.2f} \\pm {std_val:.2f}$"
                else:
                    row_str += " & - "
            else:
                row_str += " & N/A"
        row_str += r" \\"
        print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")


if __name__ == "__main__":
    generate_latex_table()
