"""
Gráficos de sumário para análise dos resultados do pipeline XAI.

Cada função gera um plot PNG em ``results/summary/`` a partir do DataFrame
consolidado de métricas (``metrics_combined.csv``).

Gráficos gerados:
    1. **metrics_by_method.png** — barplot de fidelidade agrupado por método.
    2. **model_comparison.png** — barplot comparativo ViT vs CNN.
    3. **metrics_radar.png** — barplots verticais de localidade por método.
    4. **confidence_distribution.png** — histograma de confiança separado
       por acerto/erro e modelo.
    5. **accuracy_by_class.png** — acurácia por classe de emoção e modelo.
    6. **insertion_deletion_curves.png** — curvas médias de Insertion e Deletion.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_fer.config import FIGURE_DPI


# ==============================================================================
# 1. Metrics by Method
# ==============================================================================


def _plot_metrics_by_method(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Barplot de métricas de fidelidade, agrupado por método XAI.

    **O que plota**: um gráfico de barras onde o eixo X são os métodos XAI
    (Raw, Rollout, Flow, GradCAM, etc.) e o eixo Y é o valor médio de cada
    métrica de fidelidade (AOPC_mean, AOPC_zero, Insertion_AUC, Deletion_AUC).
    Barras de erro mostram o desvio padrão.

    **Por que é útil**: permite comparar rapidamente qual método XAI produz
    heatmaps mais fiéis à decisão do modelo. AOPC alto e Insertion AUC alto
    indicam que o heatmap captura bem as regiões decisivas, enquanto
    Deletion AUC baixo confirma que removê-las degrada a predição.

    **Interpretação**: métodos com AOPC e Insertion altos e Deletion baixo
    são os mais fiéis à lógica do modelo.
    """
    metric_cols = [c for c in df.columns if c in [
        "AOPC_mean", "AOPC_zero", "Insertion_AUC", "Deletion_AUC",
    ]]
    if not metric_cols or "method" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    methods = df["method"].unique()
    x = np.arange(len(methods))
    width = 0.8 / len(metric_cols)
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    for i, col in enumerate(metric_cols):
        means = [df[df["method"] == m][col].mean() for m in methods]
        stds = [df[df["method"] == m][col].std() for m in methods]
        label = f"{col}\n(Lower is Better)" if "Deletion" in col else col
        ax.bar(x + i * width, means, width, label=label, yerr=stds,
               capsize=3, color=colors[i % len(colors)])

    ax.set_xlabel("XAI Method", fontsize=14)
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_title("Fidelity Metrics by Method", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_method.png"), dpi=FIGURE_DPI)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 2. Model Comparison
# ==============================================================================


def _plot_model_comparison(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Barplot comparativo ViT vs CNN para métricas-chave.

    **O que plota**: 4 subplots lado a lado, cada um com uma métrica
    (AOPC_mean, Insertion_AUC, Deletion_AUC, Gini), com barras para
    cada modelo (ViT em azul, CNN em vermelho) e barras de erro (std).

    **Por que é útil**: dá uma visão geral rápida de qual arquitetura
    produz explicações mais fiéis e mais concentradas. Permite identificar
    se há dominância clara de um modelo sobre o outro em termos de XAI.

    **Interpretação**: se ViT tem Insertion_AUC mais alto e Deletion_AUC
    mais baixo que CNN, seus heatmaps de atenção capturam melhor as
    regiões realmente decisivas para a classificação.
    """
    if "model" not in df.columns:
        return

    metric_cols = [c for c in ["AOPC_mean", "Insertion_AUC", "Deletion_AUC", "Gini"]
                   if c in df.columns]
    if not metric_cols:
        return

    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for i, col in enumerate(metric_cols):
        values = [df[df["model"] == m][col].mean() for m in models]
        stds = [df[df["model"] == m][col].std() for m in models]
        colors = ["#3498db", "#e74c3c"][:len(models)]
        axes[i].bar(models, values, yerr=stds, capsize=5, color=colors)
        axes[i].set_title(col, fontsize=14, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].tick_params(axis="both", which="major", labelsize=12)

    fig.suptitle("Comparison ViT vs CNN", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=FIGURE_DPI)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 3. Locality Metrics (vertical bar charts)
# ==============================================================================


def _plot_locality_metrics(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Barplots verticais de métricas de localidade, agrupados por método.

    **O que plota**: N subplots verticais (um por métrica de localidade:
    Concentration AUC, Entropy, Area@50), com barras para cada método XAI
    ordenados por modelo (CNN à esquerda, ViT à direita). Paleta de cores
    diferenciada (azuis frios para CNN, laranjas quentes para ViT) com
    divisor vertical entre os dois grupos.

    **Por que é útil**: mostra se os heatmaps são focados (baixo Area@50,
    alto Concentration_AUC) ou dispersos (alta Entropy). Essencial para
    o paper — métodos com heatmaps mais concentrados são mais interpretáveis
    para o ser humano.

    **Interpretação**:
    - Concentration AUC alto = massa concentrada em poucas regiões.
    - Entropy baixo = heatmap determinístico, não ruidoso.
    - Area@50 baixo = 50% da atenção cabe em uma fração pequena da imagem.
    """
    if "MPL_AUC" in df.columns and "Concentration_AUC" not in df.columns:
        df["Concentration_AUC"] = df["MPL_AUC"]

    if "method" not in df.columns:
        return

    metrics = [m for m in ["Concentration_AUC", "Entropy", "Area@50"] if m in df.columns]
    if not metrics:
        return

    cnn_methods = ["GradCAM", "GradCAM++", "LayerCAM"]
    vit_methods = ["Raw", "Rollout", "Flow"]

    available_methods = df["method"].unique()
    ordered_methods = ([m for m in cnn_methods if m in available_methods]
                       + [m for m in vit_methods if m in available_methods])
    others = [m for m in available_methods if m not in ordered_methods]
    ordered_methods.extend(others)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    # Paletas de cor: frias (CNN) e quentes (ViT)
    color_map = {}
    cnn_palette = matplotlib.colormaps["Blues"](np.linspace(0.5, 0.9, len(cnn_methods)))
    vit_palette = matplotlib.colormaps["Oranges"](np.linspace(0.5, 0.9, len(vit_methods)))
    for i, m in enumerate(cnn_methods):
        color_map[m] = cnn_palette[i] if i < len(cnn_palette) else "blue"
    for i, m in enumerate(vit_methods):
        color_map[m] = vit_palette[i] if i < len(vit_palette) else "orange"
    bar_colors = [color_map.get(m, "gray") for m in ordered_methods]

    for i, col in enumerate(metrics):
        ax = axes[i]
        means = [df[df["method"] == m][col].mean() for m in ordered_methods]
        stds = [df[df["method"] == m][col].std() for m in ordered_methods]

        x = np.arange(len(ordered_methods))
        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors,
               alpha=0.9, edgecolor="black", linewidth=0.5)

        split_index = len([m for m in cnn_methods if m in ordered_methods])
        if 0 < split_index < len(ordered_methods):
            ax.axvline(x=split_index - 0.5, color="gray", linestyle="--",
                       alpha=0.7, linewidth=1.5)
            if i == 0:
                y_max = ax.get_ylim()[1]
                ax.text((split_index - 1) / 2, y_max * 0.95, "CNN Models",
                        ha="center", fontweight="bold", color="#2c3e50")
                ax.text(split_index + (len(ordered_methods) - split_index - 1) / 2,
                        y_max * 0.95, "ViT Models",
                        ha="center", fontweight="bold", color="#d35400")

        if col == "Concentration_AUC":
            title = "Concentration AUC\n(Higher = More Concentrated)"
            ax.set_ylim(0, 1.05)
        elif col == "Entropy":
            title = "Entropy\n(Lower = Less Noisy)"
        elif col == "Area@50":
            title = "Area@50\n(Lower = More Localized)"
            ax.set_ylim(0, 0.7)
        else:
            title = col

        ax.set_ylabel(title.split("\n")[0], fontsize=14)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_methods, rotation=45, ha="right")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(axis="y", alpha=0.3, which="major", linestyle="-")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_radar.png"),
                dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 4. Confidence Distribution
# ==============================================================================


def _plot_confidence_distribution(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Histograma de confiança separado por acerto/erro e modelo.

    **O que plota**: histogramas sobrepostos (acerto em verde, erro em
    vermelho) para cada modelo, mostrando a distribuição de confiança
    das predições.

    **Por que é útil**: revela se o modelo é bem calibrado. Num modelo
    ideal, erros deveriam ter confiança baixa e acertos confiança alta.
    Se houver muitos erros com confiança alta, o modelo está mal calibrado
    (overconfident).

    **Interpretação**: sobreposição grande entre as cores indica
    calibração ruim; separação limpa indica bom threshold de decisão.
    """
    if "conf" not in df.columns or "correct" not in df.columns:
        return

    df_unique = df.groupby(["image_idx", "model"]).agg({
        "conf": "first", "correct": "first",
    }).reset_index()

    models = df_unique["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df_unique[df_unique["model"] == model]
        correct_conf = model_df[model_df["correct"]]["conf"]
        error_conf = model_df[~model_df["correct"]]["conf"]

        ax.hist(correct_conf, bins=20, alpha=0.7,
                label=f"Correct (n={len(correct_conf)})", color="#2ecc71")
        ax.hist(error_conf, bins=20, alpha=0.7,
                label=f"Error (n={len(error_conf)})", color="#e74c3c")

        ax.set_xlabel("Confidence", fontsize=14)
        ax.set_title(f"{model}", fontsize=16, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)

    axes[0].set_ylabel("Frequency", fontsize=14)
    fig.suptitle("Confidence Distribution by Result",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"),
                dpi=FIGURE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 5. Accuracy by Class
# ==============================================================================


def _plot_accuracy_by_class(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Barplot de acurácia por classe de emoção, separado por modelo.

    **O que plota**: barras agrupadas com ViT e CNN lado a lado para cada
    classe de emoção (angry, disgust, fear, happy, neutral, sad, surprise).
    Valores percentuais são anotados no topo de cada barra e uma linha
    pontilhada horizontal marca a média geral (overall accuracy).

    **Por que é útil**: revela quais emoções cada modelo classifica melhor
    ou pior. Classes como 'disgust' (poucas amostras) e 'fear' (confusão
    com 'surprise') costumam ter acurácia menor — isso ajuda a interpretar
    os heatmaps XAI nessas classes.

    **Interpretação**: classes com acurácia muito abaixo da média indicam
    problemas (confusão, subrepresentação). Comparar ViT vs CNN por classe
    mostra se alguma arquitetura lida melhor com emoções específicas.
    """
    if not all(c in df.columns for c in ["label", "correct", "model"]):
        return

    df_unique = df.groupby(["image_idx", "model"]).agg({
        "label": "first", "correct": "first",
    }).reset_index()

    models = sorted(df_unique["model"].unique())
    classes = sorted(df_unique["label"].unique())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    colors = {"VIT": "#3498db", "CNN": "#e74c3c", "ViT": "#3498db"}

    for i, model in enumerate(models):
        model_df = df_unique[df_unique["model"] == model]
        accuracies = [
            model_df[model_df["label"] == c]["correct"].mean()
            if len(model_df[model_df["label"] == c]) > 0 else 0
            for c in classes
        ]

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, accuracies, width, label=model,
                      color=colors.get(model, f"C{i}"), alpha=0.85)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.0%}", ha="center", fontsize=8, fontweight="bold")

    overall_mean = df_unique["correct"].mean()
    ax.axhline(y=overall_mean, color="gray", linestyle="--",
               label=f"Overall Mean: {overall_mean:.1%}")

    ax.set_xlabel("Class", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Accuracy by Emotion Class (per Model)",
                 fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_class.png"), dpi=FIGURE_DPI)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# 6. Insertion / Deletion Curves
# ==============================================================================


def _plot_insertion_deletion_curves(
    df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Curvas médias de Insertion e Deletion para cada (modelo, método).

    **O que plota**: dois subplots lado a lado:
    - **Insertion** (esquerda): curva da confiança à medida que pixels
      mais relevantes são inseridos na baseline. Linhas mais altas = melhor.
    - **Deletion** (direita): curva da confiança à medida que pixels mais
      relevantes são removidos. Linhas mais baixas = melhor.

    **Por que é útil**: é o gráfico central do paper para demonstrar
    fidelidade. Mostra visualmente como a confiança do modelo responde
    à remoção/inserção progressiva das regiões destacadas pelo heatmap.
    AUC dessas curvas são INSERTION_AUC e DELETION_AUC (métricas numéricas).

    **Interpretação**: um bom heatmap produz curva de Insertion que sobe
    rapidamente (os primeiros pixels já restauram a confiança) e curva de
    Deletion que cai rapidamente (remover poucos pixels já degrada).
    """
    if not all(c in df.columns for c in ["insertion_confs", "deletion_confs"]):
        return

    def parse_array(x):
        if isinstance(x, str):
            return np.array(eval(x))
        return np.array(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    groups = df.groupby(["model", "method"])
    for i, ((model_name, method_name), group) in enumerate(groups):
        ins_curves = np.stack([parse_array(x) for x in group["insertion_confs"].values])
        del_curves = np.stack([parse_array(x) for x in group["deletion_confs"].values])

        ins_mean = ins_curves.mean(axis=0)
        del_mean = del_curves.mean(axis=0)
        steps = np.linspace(0, 1, len(ins_mean))

        label = f"{model_name}-{method_name}"
        color = colors[i % len(colors)]

        axes[0].plot(steps, ins_mean, label=label, color=color, linewidth=2)
        axes[1].plot(steps, del_mean, label=label, color=color, linewidth=2,
                     linestyle="--")

    axes[0].set_title("Insertion Curve (Higher is Better)",
                      fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Fraction of Pixels Inserted", fontsize=14)
    axes[0].set_ylabel("Class Probability", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].tick_params(axis="both", labelsize=12)

    axes[1].set_title("Deletion Curve (Lower is Better)",
                      fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Fraction of Pixels Removed", fontsize=14)
    axes[1].set_ylabel("Class Probability", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "insertion_deletion_curves.png"),
                dpi=FIGURE_DPI)
    if show:
        plt.show()
    else:
        plt.close()


# ==============================================================================
# Função pública
# ==============================================================================


def generate_all_summary_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    show: bool = False,
) -> None:
    """Gera todos os gráficos de sumário e salva em ``output_dir``.

    Chamada principal pelo pipeline após consolidar os resultados.
    """
    os.makedirs(output_dir, exist_ok=True)

    _plot_metrics_by_method(results_df, output_dir, show)
    _plot_confidence_distribution(results_df, output_dir, show)
    _plot_accuracy_by_class(results_df, output_dir, show)
    _plot_model_comparison(results_df, output_dir, show)
    _plot_locality_metrics(results_df, output_dir, show)

    try:
        _plot_insertion_deletion_curves(results_df, output_dir, show)
    except Exception as e:
        print(f"Erro ao plotar curvas Insertion/Deletion: {e}")
