"""
Entry point do pacote xai_fer.

Permite executar o pipeline via ``python -m xai_fer`` com suporte a:
    - Menu interativo (sem argumentos)
    - CLI com flags (``--n_samples``, ``--models``, ``--plots-only``, etc.)

Orquestra o pipeline completo em passes:
    1. Pass 1: executa métodos XAI nativos (atenção/CAM) e calcula métricas.
    2. Pass 2 (se estratificado): seleciona subconjunto e gera heatmaps.
    3. Pass 2.5 (opcional): LIME/SHAP nas imagens estratificadas.
    4. Consolidação: salva CSV, gera gráficos e relatórios de análise.
"""

import os
import sys
import time
import argparse

import pandas as pd

from xai_fer.config import (
    N_SAMPLES,
    N_SAMPLES_AGNOSTIC,
    RESULTS_DIR,
    SUMMARY_DIR,
    HEATMAPS_DIR,
    VIT_XAI_METHODS,
    CNN_XAI_METHODS,
    AGNOSTIC_XAI_METHODS,
    HEATMAP_SELECTION_STRATEGY,
    create_results_dirs,
    print_config,
)
from xai_fer.data.loader import load_dataset, get_all_images
from xai_fer.visualization.plots import generate_all_summary_plots
from xai_fer.evaluation.analysis import generate_analysis_report
from xai_fer.pipeline.runner import XAIPipelineRunner

# Imports dos modelos (lazy — só falham se realmente usados)
try:
    from xai_fer.models.vit_explainer import (
        load_vit_model, build_transform_from_convnext, run_xai_on_image,
    )
    from xai_fer.models.cnn_explainer import (
        load_cnn_model, build_cnn_transform, run_xai_on_image_cnn,
    )
    from xai_fer.models.agnostic_explainer import run_agnostic_xai
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: {e}")
    sys.exit(1)


# ==============================================================================
# Pipeline Principal
# ==============================================================================


def run_full_analysis(
    n_samples: int = None,
    n_samples_agnostic: int = None,
    models: list = None,
    run_agnostic: bool = False,
    save_heatmaps: bool = True,
    generate_plots: bool = True,
    generate_analysis: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Executa análise XAI completa (Pass 1 → Seleção → Pass 2 → Plots)."""
    start_time = time.time()
    create_results_dirs()

    if n_samples is None:
        n_samples = N_SAMPLES
    if n_samples_agnostic is None:
        n_samples_agnostic = N_SAMPLES_AGNOSTIC
    if models is None:
        models = ["vit", "cnn"]

    if verbose:
        print("\n" + "=" * 60)
        print("XAI ANALYSIS — FULL PIPELINE")
        print("=" * 60)
        print_config()

    # ── Pass 1: métricas ─────────────────────────────────────────────────────
    pass1_save_heatmaps = save_heatmaps
    if HEATMAP_SELECTION_STRATEGY == "stratified":
        pass1_save_heatmaps = False
        if verbose:
            print("\n[STRATEGY] Modo Estratificado: Pass 1 (Métricas) → Seleção → Pass 2 (Heatmaps)")

    all_results = []
    runners = {}

    if "vit" in models:
        if verbose:
            print(f"\n[ViT] Inicializando pipeline...")
        model, cfg, device = load_vit_model()
        transform, _, _, _ = build_transform_from_convnext()

        runner = XAIPipelineRunner(
            model=model, transform=transform, device=device,
            model_type="vit", xai_methods=tuple(VIT_XAI_METHODS),
            xai_function=run_xai_on_image,
        )
        runners["vit"] = runner

        if verbose:
            print(f"[ViT] Processando Pass 1 ({n_samples} imagens)...")
        df = load_dataset(n_samples=n_samples)
        if not df.empty:
            res = runner.run(df, save_heatmaps=pass1_save_heatmaps, save_metrics=True, verbose=verbose)
            all_results.append(res)

    if "cnn" in models:
        if verbose:
            print(f"\n[CNN] Inicializando pipeline...")
        model, data_config, device = load_cnn_model()
        transform, _, _, _ = build_cnn_transform(data_config)

        runner = XAIPipelineRunner(
            model=model, transform=transform, device=device,
            model_type="cnn", xai_methods=tuple(CNN_XAI_METHODS),
            xai_function=run_xai_on_image_cnn,
        )
        runners["cnn"] = runner

        if verbose:
            print(f"[CNN] Processando Pass 1 ({n_samples} imagens)...")
        df = load_dataset(n_samples=n_samples)
        if not df.empty:
            res = runner.run(df, save_heatmaps=pass1_save_heatmaps, save_metrics=True, verbose=verbose)
            all_results.append(res)

    combined_df = pd.concat([d for d in all_results if not d.empty], ignore_index=True)

    # ── Pass 2: seleção estratificada + heatmaps ─────────────────────────────
    if not combined_df.empty and HEATMAP_SELECTION_STRATEGY == "stratified":
        from xai_fer.pipeline.selector import StratifiedSelector

        if verbose:
            print("\n" + "=" * 60)
            print("PASS 2: SELEÇÃO ESTRATIFICADA")
            print("=" * 60)

        selector = StratifiedSelector(combined_df)
        selected_df = selector.select_candidates()

        if not selected_df.empty:
            selection_csv = os.path.join(RESULTS_DIR, "heatmap_selection.csv")
            selected_df[
                ["image_idx", "filename", "label", "model", "conf", "correct", "selection_reason"]
            ].to_csv(selection_csv, index=False)
            if verbose:
                print(f"Lista salva em: {selection_csv}")

            target_paths = selected_df["path"].unique().tolist()
            if verbose:
                print(f"Gerando heatmaps para {len(target_paths)} imagens selecionadas...")

            df_subset = load_dataset(target_paths=target_paths)

            for model_name, runner in runners.items():
                if verbose:
                    print(f"[{model_name.upper()}] Gerando heatmaps Pass 2...")
                runner.run(df_subset, save_heatmaps=True, save_metrics=False, verbose=verbose)

            # ── Pass 2.5: LIME/SHAP ──────────────────────────────────────────
            if run_agnostic and AGNOSTIC_XAI_METHODS:
                from xai_fer.visualization.heatmaps import save_xai_visualization
                from xai_fer.utils import get_label_name

                n_stratified = len(df_subset)
                n_agnostic = min(n_samples_agnostic, n_stratified)

                if n_samples_agnostic > n_stratified:
                    print(f"\n[AGNOSTIC] N_SAMPLES_AGNOSTIC={n_samples_agnostic} > "
                          f"estratificadas={n_stratified}. Limitando a {n_stratified}.")

                df_agnostic = df_subset.head(n_agnostic)

                for model_name, runner in runners.items():
                    if verbose:
                        print(f"\n[AGNOSTIC-{model_name.upper()}] LIME/SHAP em {n_agnostic} imagens...")

                    agnostic_dir = os.path.join(HEATMAPS_DIR, "agnostic", model_name)
                    os.makedirs(agnostic_dir, exist_ok=True)

                    for i, (idx, row) in enumerate(df_agnostic.iterrows()):
                        if verbose:
                            print(f"  [{i+1}/{n_agnostic}] LIME/SHAP ({model_name}): {row['filename']}...")
                        try:
                            pil_img, pred_idx, conf, agnostic_maps = run_agnostic_xai(
                                row["path"], runner.model, runner.device,
                                model_type=model_name, methods=tuple(AGNOSTIC_XAI_METHODS),
                            )
                            pred_label = get_label_name(pred_idx)
                            status_str = "OK" if pred_label == row["label"] else "ERR"
                            hm_filename = f"{idx:03d}_{status_str}_{row['label']}_agnostic.png"
                            hm_path = os.path.join(agnostic_dir, hm_filename)
                            title = (f"AGNOSTIC-{model_name.upper()} | {status_str} | "
                                     f"True={row['label']} | Pred={pred_label} | Conf={conf:.2%}")
                            save_xai_visualization(pil_img, agnostic_maps, hm_path, title, show=False)
                        except Exception as e:
                            if verbose:
                                print(f"    ERRO LIME/SHAP: {e}")

    # ── Finalização ──────────────────────────────────────────────────────────
    if not combined_df.empty:
        combined_df.to_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"), index=False)
        if generate_plots:
            if verbose:
                print("\n[PLOTS] Gerando gráficos...")
            generate_all_summary_plots(combined_df, SUMMARY_DIR, show=False)
        if generate_analysis:
            generate_analysis_report(combined_df, verbose=verbose)

    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print(f"PIPELINE CONCLUÍDO em {elapsed:.1f}s")
        print("=" * 60)

    return combined_df


# ==============================================================================
# Modos Auxiliares
# ==============================================================================


def run_agnostic_only(models: list = None, verbose: bool = True) -> None:
    """Executa apenas LIME/SHAP nas imagens já selecionadas pelo estratificado."""
    from tqdm import tqdm
    from xai_fer.visualization.heatmaps import save_xai_visualization
    from xai_fer.utils import get_label_name

    start_time = time.time()
    create_results_dirs()

    if models is None:
        models = ["vit", "cnn"]

    selection_csv = os.path.join(RESULTS_DIR, "heatmap_selection.csv")
    if not os.path.exists(selection_csv):
        print(f"ERRO: {selection_csv} não encontrado. Execute o pipeline completo primeiro.")
        return

    if verbose:
        print("\n" + "=" * 60)
        print("AGNOSTIC-ONLY MODE: LIME/SHAP")
        print("=" * 60)

    selection_df = pd.read_csv(selection_csv)
    unique_filenames = selection_df["filename"].unique().tolist()
    if verbose:
        print(f"[AGNOSTIC] {len(unique_filenames)} filenames únicos no CSV")

    all_images = get_all_images()
    filtered = [img for img in all_images if img["filename"] in unique_filenames]
    df_agnostic = pd.DataFrame(filtered)

    idx_map = selection_df.drop_duplicates("filename").set_index("filename")["image_idx"]
    df_agnostic["image_idx"] = df_agnostic["filename"].map(idx_map)
    n_images = len(df_agnostic)
    if verbose:
        print(f"[AGNOSTIC] {n_images} imagens únicas para processar")

    for model_name in models:
        if verbose:
            print(f"\n[AGNOSTIC-{model_name.upper()}] Carregando modelo...")

        if model_name == "vit":
            model, _, device = load_vit_model()
        else:
            model, _, device = load_cnn_model()

        agnostic_dir = os.path.join(HEATMAPS_DIR, "agnostic", model_name)
        os.makedirs(agnostic_dir, exist_ok=True)

        if verbose:
            print(f"[AGNOSTIC-{model_name.upper()}] LIME/SHAP em {n_images} imagens...")

        for idx, row in tqdm(df_agnostic.iterrows(), total=n_images,
                             desc=f"LIME/SHAP ({model_name.upper()})", unit="img"):
            try:
                pil_img, pred_idx, conf, agnostic_maps = run_agnostic_xai(
                    row["path"], model, device,
                    model_type=model_name, methods=tuple(AGNOSTIC_XAI_METHODS),
                )
                pred_label = get_label_name(pred_idx)
                status_str = "OK" if pred_label == row["label"] else "ERR"
                image_idx = row.get("image_idx", idx)
                hm_filename = f"{image_idx:03d}_{status_str}_{row['label']}_agnostic.png"
                hm_path = os.path.join(agnostic_dir, hm_filename)
                title = (f"AGNOSTIC-{model_name.upper()} | {status_str} | "
                         f"True={row['label']} | Pred={pred_label} | Conf={conf:.2%}")
                save_xai_visualization(pil_img, agnostic_maps, hm_path, title, show=False)
            except Exception as e:
                if verbose:
                    tqdm.write(f"  ERRO {row['filename']}: {e}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n[AGNOSTIC] Concluído em {elapsed:.1f}s")
        print(f"[AGNOSTIC] Heatmaps salvos em: {os.path.join(HEATMAPS_DIR, 'agnostic')}")


def run_plots_only(verbose: bool = True) -> None:
    """Regenera gráficos a partir do CSV existente."""
    csv_path = os.path.join(RESULTS_DIR, "metrics_combined.csv")
    if not os.path.exists(csv_path):
        print(f"ERRO: {csv_path} não encontrado. Execute o pipeline primeiro.")
        return
    if verbose:
        print("\n[PLOTS] Carregando dados existentes...")
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"[PLOTS] {len(df)} registros carregados")
        print(f"[PLOTS] Gerando gráficos em {SUMMARY_DIR}...")
    generate_all_summary_plots(df, SUMMARY_DIR, show=False)
    if verbose:
        print("[PLOTS] Gráficos regenerados com sucesso!")


def run_analysis_only(verbose: bool = True) -> None:
    """Regenera CSVs de análise a partir do CSV existente."""
    csv_path = os.path.join(RESULTS_DIR, "metrics_combined.csv")
    if not os.path.exists(csv_path):
        print(f"ERRO: {csv_path} não encontrado. Execute o pipeline primeiro.")
        return
    if verbose:
        print("\n[ANALYSIS] Carregando dados existentes...")
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"[ANALYSIS] {len(df)} registros carregados")
    generate_analysis_report(df, verbose=verbose)
    if verbose:
        print("[ANALYSIS] CSVs de análise regenerados!")


# ==============================================================================
# Menu Interativo
# ==============================================================================


def show_menu() -> None:
    """Exibe o menu principal."""
    print("\n" + "═" * 60)
    print("         XAI ANALYSIS — MENU PRINCIPAL")
    print("═" * 60)
    print()
    print("  [1] Pipeline Completo (ViT + CNN)")
    print("  [2] Pipeline Apenas ViT")
    print("  [3] Pipeline Apenas CNN")
    print("  [4] Executar LIME/SHAP (imagens estratificadas)")
    print("  [5] Regenerar Gráficos (usa CSV existente)")
    print("  [6] Regenerar CSVs de Análise")
    print("  [0] Sair")
    print()


def _prompt_int(prompt: str, default: int = None) -> int:
    while True:
        try:
            default_str = f" [{default}]" if default else ""
            user_input = input(f"{prompt}{default_str}: ").strip()
            if not user_input and default is not None:
                return default
            return int(user_input)
        except ValueError:
            print("Por favor, digite um número inteiro.")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "S/n" if default else "s/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ("s", "sim", "y", "yes")


def run_interactive_menu() -> None:
    """Executa o menu interativo."""
    while True:
        show_menu()
        try:
            choice = input("Escolha uma opção: ").strip()

            if choice == "0":
                print("\nAté logo!")
                break
            elif choice == "1":
                n = _prompt_int("Número de imagens para análise", N_SAMPLES)
                agnostic = _prompt_yes_no("Executar LIME/SHAP também?", False)
                n_agnostic = None
                if agnostic:
                    n_agnostic = _prompt_int("Número de imagens para LIME/SHAP", N_SAMPLES_AGNOSTIC)
                run_full_analysis(n_samples=n, n_samples_agnostic=n_agnostic,
                                  models=["vit", "cnn"], run_agnostic=agnostic, verbose=True)
            elif choice == "2":
                n = _prompt_int("Número de imagens para análise", N_SAMPLES)
                run_full_analysis(n_samples=n, models=["vit"], verbose=True)
            elif choice == "3":
                n = _prompt_int("Número de imagens para análise", N_SAMPLES)
                run_full_analysis(n_samples=n, models=["cnn"], verbose=True)
            elif choice == "4":
                models_input = input("Modelos (vit, cnn ou ambos) [vit cnn]: ").strip()
                run_agnostic_only(models=models_input.split() if models_input else ["vit", "cnn"],
                                  verbose=True)
            elif choice == "5":
                run_plots_only(verbose=True)
            elif choice == "6":
                run_analysis_only(verbose=True)
            else:
                print("Opção inválida. Tente novamente.")

        except KeyboardInterrupt:
            print("\n\nOperação cancelada.")
            break
        except Exception as e:
            print(f"\nErro: {e}")


# ==============================================================================
# CLI
# ==============================================================================


def main():
    """Função principal com suporte a CLI e menu interativo."""
    if len(sys.argv) == 1:
        run_interactive_menu()
        return

    parser = argparse.ArgumentParser(
        description="XAI Analysis Pipeline para Reconhecimento de Emoções",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python -m xai_fer                          # Menu interativo
  python -m xai_fer --n_samples 100          # Pipeline completo com 100 imagens
  python -m xai_fer --models vit             # Apenas ViT
  python -m xai_fer --agnostic-only          # Apenas LIME/SHAP (usa CSV existente)
  python -m xai_fer --plots-only             # Regenerar gráficos
  python -m xai_fer --n_samples 50 --agnostic --n_agnostic 10
        """,
    )

    parser.add_argument("--n_samples", "-n", type=int, default=None,
                        help=f"Imagens para análise (default: {N_SAMPLES})")
    parser.add_argument("--n_agnostic", type=int, default=None,
                        help=f"Imagens para LIME/SHAP (default: {N_SAMPLES_AGNOSTIC})")
    parser.add_argument("--models", "-m", nargs="+", default=["vit", "cnn"],
                        choices=["vit", "cnn"], help="Modelos a processar")
    parser.add_argument("--agnostic", "-a", action="store_true",
                        help="Executar LIME/SHAP nas imagens estratificadas")
    parser.add_argument("--agnostic-only", action="store_true",
                        help="Executar APENAS LIME/SHAP (usa CSV existente)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerar apenas gráficos")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Regenerar apenas CSVs de análise")
    parser.add_argument("--no-heatmaps", action="store_true",
                        help="Não salvar heatmaps visuais")
    parser.add_argument("--no-plots", action="store_true",
                        help="Não gerar gráficos de resumo")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Não gerar CSVs de análise")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Modo silencioso")

    args = parser.parse_args()

    if args.plots_only:
        run_plots_only(verbose=not args.quiet)
    elif args.analysis_only:
        run_analysis_only(verbose=not args.quiet)
    elif args.agnostic_only:
        run_agnostic_only(models=args.models, verbose=not args.quiet)
    else:
        run_full_analysis(
            n_samples=args.n_samples, n_samples_agnostic=args.n_agnostic,
            models=args.models, run_agnostic=args.agnostic,
            save_heatmaps=not args.no_heatmaps, generate_plots=not args.no_plots,
            generate_analysis=not args.no_analysis, verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
