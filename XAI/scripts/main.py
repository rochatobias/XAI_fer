# ==============================================================================
# Main - Pipeline Principal XAI
# ==============================================================================
# Execute este arquivo para rodar análise XAI com ViT e/ou CNN
# Use --help para ver opções de linha de comando
# ==============================================================================

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

from config import (
    N_SAMPLES, N_SAMPLES_AGNOSTIC, RESULTS_DIR, HEATMAPS_DIR, SUMMARY_DIR,
    VIT_XAI_METHODS, CNN_XAI_METHODS, AGNOSTIC_XAI_METHODS,
    MEAN, STD, create_results_dirs, print_config, get_device
)
from data_loader import load_dataset
from utils import get_label_name


def run_vit_pipeline(
    n_samples: int = None,
    save_heatmaps: bool = True,
    save_metrics: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Pipeline XAI para modelo ViT (attention-based).
    
    Args:
        n_samples: Número de imagens a processar
        save_heatmaps: Salvar visualizações
        save_metrics: Salvar CSV de métricas
        verbose: Imprimir progresso
    
    Returns:
        DataFrame com resultados
    """
    from vit import load_vit_model, build_transform_from_convnext, run_xai_on_image
    from metrics import compute_all_metrics
    from visualization import save_xai_visualization
    
    if n_samples is None:
        n_samples = N_SAMPLES
    
    if verbose:
        print("\n[ViT] Carregando modelo...")
    model, cfg, device = load_vit_model()
    transform, img_size, mean, std = build_transform_from_convnext()
    
    if verbose:
        print(f"[ViT] Carregando dataset ({n_samples} imagens)...")
    df = load_dataset(n_samples=n_samples)
    if len(df) == 0:
        print("ERRO: Nenhuma imagem encontrada!")
        return pd.DataFrame()
    
    heatmaps_dir = os.path.join(HEATMAPS_DIR, "vit")
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    all_results = []
    for idx, row in df.iterrows():
        img_path = row['path']
        true_label = row['label']
        true_label_idx = row['label_idx']
        filename = row['filename']
        
        if verbose:
            print(f"\n  [{idx+1}/{len(df)}] {filename} (Label: {true_label})")
        
        try:
            pil_img, pred_idx, conf, maps = run_xai_on_image(
                img_path, model, transform, device, methods=tuple(VIT_XAI_METHODS)
            )
            pred_label = get_label_name(pred_idx)
            correct = (pred_idx == true_label_idx)
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"       Pred: {pred_label} ({conf:.2%}) {status}")
            
            # Salvar visualização
            if save_heatmaps:
                status_str = "OK" if correct else "ERR"
                heatmap_path = os.path.join(
                    heatmaps_dir, 
                    f"{idx:03d}_{status_str}_{true_label}_pred{pred_label}_{conf:.2f}.png"
                )
                title = f"{status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                save_xai_visualization(pil_img, maps, heatmap_path, title, show=False)
            
            # Calcular métricas para cada método XAI
            for method_name, heatmap in maps.items():
                if verbose:
                    print(f"       Métricas: {method_name}...")
                metrics = compute_all_metrics(
                    model, pil_img, heatmap, device, mean, std,
                    model_type="vit", true_label_idx=true_label_idx
                )
                result = {
                    'model': 'ViT',
                    'image_idx': idx,
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
    
    if save_metrics and len(results_df) > 0:
        csv_path = os.path.join(RESULTS_DIR, "metrics_vit.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[ViT] Métricas salvas em: {csv_path}")
    
    return results_df


def run_cnn_pipeline(
    n_samples: int = None,
    save_heatmaps: bool = True,
    save_metrics: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Pipeline XAI para modelo CNN (CAM-based).
    """
    from cnn import load_cnn_model, build_cnn_transform, run_xai_on_image_cnn
    from metrics import compute_all_metrics
    from visualization import save_xai_visualization
    
    if n_samples is None:
        n_samples = N_SAMPLES
    
    if verbose:
        print("\n[CNN] Carregando modelo...")
    model, data_config, device = load_cnn_model()
    transform, img_size, mean, std = build_cnn_transform(data_config)
    
    if verbose:
        print(f"[CNN] Carregando dataset ({n_samples} imagens)...")
    df = load_dataset(n_samples=n_samples)
    if len(df) == 0:
        print("ERRO: Nenhuma imagem encontrada!")
        return pd.DataFrame()
    
    heatmaps_dir = os.path.join(HEATMAPS_DIR, "cnn")
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    all_results = []
    for idx, row in df.iterrows():
        img_path = row['path']
        true_label = row['label']
        true_label_idx = row['label_idx']
        filename = row['filename']
        
        if verbose:
            print(f"\n  [{idx+1}/{len(df)}] {filename} (Label: {true_label})")
        
        try:
            pil_img, pred_idx, conf, maps = run_xai_on_image_cnn(
                img_path, model, transform, device, methods=tuple(CNN_XAI_METHODS)
            )
            pred_label = get_label_name(pred_idx)
            correct = (pred_idx == true_label_idx)
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"       Pred: {pred_label} ({conf:.2%}) {status}")
            
            # Salvar visualização
            if save_heatmaps:
                status_str = "OK" if correct else "ERR"
                heatmap_path = os.path.join(
                    heatmaps_dir,
                    f"{idx:03d}_{status_str}_{true_label}_pred{pred_label}_{conf:.2f}.png"
                )
                title = f"CNN | {status_str} | True={true_label} | Pred={pred_label}"
                save_xai_visualization(pil_img, maps, heatmap_path, title, show=False)
            
            # Calcular métricas
            for method_name, heatmap in maps.items():
                if verbose:
                    print(f"       Métricas: {method_name}...")
                metrics = compute_all_metrics(
                    model, pil_img, heatmap, device, mean, std,
                    model_type="cnn", true_label_idx=true_label_idx
                )
                result = {
                    'model': 'CNN',
                    'image_idx': idx,
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
    
    if save_metrics and len(results_df) > 0:
        csv_path = os.path.join(RESULTS_DIR, "metrics_cnn.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[CNN] Métricas salvas em: {csv_path}")
    
    return results_df


def run_agnostic_pipeline(
    n_samples: int = None,
    model_type: str = "vit",
    save_heatmaps: bool = True,
    save_metrics: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Pipeline XAI com métodos agnósticos (LIME/SHAP).
    
    Usa N_SAMPLES_AGNOSTIC por padrão (métodos são mais lentos).
    """
    from agnostic import run_agnostic_xai
    from metrics import compute_all_metrics
    from visualization import save_xai_visualization
    
    if n_samples is None:
        n_samples = N_SAMPLES_AGNOSTIC
    
    # Carrega modelo apropriado
    if model_type == "vit":
        from vit import load_vit_model, build_transform_from_convnext
        model, cfg, device = load_vit_model()
        transform, img_size, mean, std = build_transform_from_convnext()
    else:
        from cnn import load_cnn_model, build_cnn_transform
        model, data_config, device = load_cnn_model()
        transform, img_size, mean, std = build_cnn_transform(data_config)
    
    if verbose:
        print(f"\n[Agnostic/{model_type.upper()}] Carregando dataset ({n_samples} imagens)...")
    df = load_dataset(n_samples=n_samples)
    if len(df) == 0:
        return pd.DataFrame()
    
    heatmaps_dir = os.path.join(HEATMAPS_DIR, f"agnostic_{model_type}")
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    all_results = []
    for idx, row in df.iterrows():
        img_path = row['path']
        true_label = row['label']
        true_label_idx = row['label_idx']
        filename = row['filename']
        
        if verbose:
            print(f"\n  [{idx+1}/{len(df)}] {filename}")
        
        try:
            pil_img, pred_idx, conf, maps = run_agnostic_xai(
                img_path, model, device, model_type=model_type,
                methods=tuple(AGNOSTIC_XAI_METHODS)
            )
            pred_label = get_label_name(pred_idx)
            correct = (pred_idx == true_label_idx)
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"       Pred: {pred_label} ({conf:.2%}) {status}")
            
            if save_heatmaps:
                status_str = "OK" if correct else "ERR"
                heatmap_path = os.path.join(
                    heatmaps_dir,
                    f"{idx:03d}_{status_str}_{true_label}_{model_type}.png"
                )
                save_xai_visualization(pil_img, maps, heatmap_path, show=False)
            
            for method_name, heatmap in maps.items():
                if verbose:
                    print(f"       Métricas: {method_name}...")
                metrics = compute_all_metrics(
                    model, pil_img, heatmap, device, mean, std,
                    model_type=model_type, true_label_idx=true_label_idx
                )
                result = {
                    'model': f'{model_type.upper()}-Agnostic',
                    'image_idx': idx,
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
    
    if save_metrics and len(results_df) > 0:
        csv_path = os.path.join(RESULTS_DIR, f"metrics_agnostic_{model_type}.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[Agnostic] Métricas salvas em: {csv_path}")
    
    return results_df


def run_full_analysis(
    n_samples: int = None,
    n_samples_agnostic: int = None,
    models: list = None,
    run_agnostic: bool = False,
    save_heatmaps: bool = True,
    generate_plots: bool = True,
    generate_analysis: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Executa análise XAI completa com todos os modelos selecionados.
    
    Args:
        n_samples: Número de imagens para métodos nativos
        n_samples_agnostic: Número de imagens para LIME/SHAP
        models: Lista de modelos ['vit', 'cnn'] (default: ambos)
        run_agnostic: Se True, roda também LIME/SHAP
        save_heatmaps: Salvar visualizações
        generate_plots: Gerar gráficos de sumário
        generate_analysis: Gerar CSVs de análise para pesquisa
        verbose: Imprimir progresso
    
    Returns:
        DataFrame consolidado com todos os resultados
    """
    from visualization import generate_all_summary_plots
    from analysis import generate_analysis_report
    
    start_time = time.time()
    
    if n_samples is None:
        n_samples = N_SAMPLES
    if n_samples_agnostic is None:
        n_samples_agnostic = N_SAMPLES_AGNOSTIC
    if models is None:
        models = ["vit", "cnn"]
    
    if verbose:
        print("\n" + "=" * 60)
        print("XAI ANALYSIS - FULL PIPELINE")
        print("=" * 60)
        print_config()
    
    create_results_dirs()
    
    all_results = []
    
    # Pipelines nativos
    if "vit" in models:
        if verbose:
            print("\n" + "-" * 40)
            print("PIPELINE ViT (Attention-based XAI)")
            print("-" * 40)
        vit_results = run_vit_pipeline(n_samples, save_heatmaps, save_metrics=True, verbose=verbose)
        all_results.append(vit_results)
    
    if "cnn" in models:
        if verbose:
            print("\n" + "-" * 40)
            print("PIPELINE CNN (CAM-based XAI)")
            print("-" * 40)
        cnn_results = run_cnn_pipeline(n_samples, save_heatmaps, save_metrics=True, verbose=verbose)
        all_results.append(cnn_results)
    
    # Pipelines agnósticos (opcional)
    if run_agnostic:
        if verbose:
            print("\n" + "-" * 40)
            print("PIPELINE AGNOSTIC (LIME/SHAP)")
            print("-" * 40)
        
        for model_type in models:
            agnostic_results = run_agnostic_pipeline(
                n_samples_agnostic, model_type, save_heatmaps, 
                save_metrics=True, verbose=verbose
            )
            all_results.append(agnostic_results)
    
    # Consolida resultados
    combined_df = pd.concat([df for df in all_results if len(df) > 0], ignore_index=True)
    
    if len(combined_df) > 0:
        # Salva CSV consolidado
        csv_path = os.path.join(RESULTS_DIR, "metrics_combined.csv")
        combined_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[COMBINED] Métricas consolidadas: {csv_path}")
        
        # Gera gráficos
        if generate_plots:
            if verbose:
                print("\n[PLOTS] Gerando gráficos de sumário...")
            generate_all_summary_plots(combined_df, SUMMARY_DIR, show=False)
        
        # Gera CSVs de análise
        if generate_analysis:
            generate_analysis_report(combined_df, verbose=verbose)
    
    # Resumo final
    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print("RESUMO DA ANÁLISE")
        print("=" * 60)
        print(f"Tempo total: {elapsed:.1f}s")
        print(f"Total de registros: {len(combined_df)}")
        if len(combined_df) > 0:
            print(f"\nAcurácia geral:")
            for model_name in combined_df["model"].unique():
                model_df = combined_df[combined_df["model"] == model_name]
                acc = model_df.groupby("image_idx")["correct"].first().mean()
                n_imgs = len(model_df["image_idx"].unique())
                print(f"  {model_name}: {acc:.1%} ({n_imgs} imagens)")
        print("=" * 60)
    
    return combined_df


def main():
    """Função principal com suporte a argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="XAI Analysis Pipeline")
    parser.add_argument("--n_samples", type=int, default=None, help="Número de imagens")
    parser.add_argument("--n_agnostic", type=int, default=None, help="Número para LIME/SHAP")
    parser.add_argument("--models", nargs="+", default=["vit", "cnn"], help="Modelos: vit, cnn")
    parser.add_argument("--agnostic", action="store_true", help="Incluir LIME/SHAP")
    parser.add_argument("--no-heatmaps", action="store_true", help="Não salvar heatmaps")
    parser.add_argument("--no-plots", action="store_true", help="Não gerar gráficos")
    parser.add_argument("--no-analysis", action="store_true", help="Não gerar CSVs de análise")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  XAI para Classificação de Emoções".center(56) + "  #")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    results = run_full_analysis(
        n_samples=args.n_samples,
        n_samples_agnostic=args.n_agnostic,
        models=args.models,
        run_agnostic=args.agnostic,
        save_heatmaps=not args.no_heatmaps,
        generate_plots=not args.no_plots,
        generate_analysis=not args.no_analysis,
        verbose=not args.quiet
    )
    
    if len(results) > 0:
        print("\n✓ Pipeline concluído com sucesso!")
        print(f"\nResultados em: {RESULTS_DIR}")
    else:
        print("\n✗ Pipeline falhou - verifique os erros acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
