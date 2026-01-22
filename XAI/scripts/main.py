"""Main - Pipeline Principal XAI.

Execute: python main.py --help para ver opções.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

from config import (
    N_SAMPLES, N_SAMPLES_AGNOSTIC, RESULTS_DIR, SUMMARY_DIR, HEATMAPS_DIR,
    VIT_XAI_METHODS, CNN_XAI_METHODS, AGNOSTIC_XAI_METHODS,
    HEATMAP_SELECTION_STRATEGY, create_results_dirs, print_config
)
from data_loader import load_dataset
from visualization import generate_all_summary_plots
from analysis import generate_analysis_report
from pipeline_runner import XAIPipelineRunner

# Imports dos modelos e funções XAI
try:
    from vit import load_vit_model, build_transform_from_convnext, run_xai_on_image
    from cnn import load_cnn_model, build_cnn_transform, run_xai_on_image_cnn
    from agnostic import run_agnostic_xai
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: {e}")
    sys.exit(1)


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
    """Executa análise XAI completa."""
    
    start_time = time.time()
    create_results_dirs()
    
    if n_samples is None: n_samples = N_SAMPLES
    if n_samples_agnostic is None: n_samples_agnostic = N_SAMPLES_AGNOSTIC
    if models is None: models = ["vit", "cnn"]
    
    if verbose:
        print("\n" + "=" * 60)
        print("XAI ANALYSIS - FULL PIPELINE")
        print("=" * 60)
        print_config()
        
    # ==========================================================================
    # 1. Configuração da Estratégia de Execução (Pass 1 / Pass 2)
    # ==========================================================================
    
    # Se estratificado, Pass 1 NÃO salva heatmaps nem calcula métricas pesadas se não precisasse
    # Mas aqui queremos métricas para selecionar.
    pass1_save_heatmaps = save_heatmaps
    if HEATMAP_SELECTION_STRATEGY == "stratified":
        pass1_save_heatmaps = False
        if verbose:
            print("\n[STRATEGY] Modo Estratificado Ativo: Pass 1 (Métricas) -> Seleção -> Pass 2 (Heatmaps)")

    # ==========================================================================
    # 2. Execução do Pass 1 (Métricas para todos ou N amostras)
    # ==========================================================================
    
    all_results = []
    runners = {}
    
    # Carregamento de dados (feito uma vez se possível, mas transform pode variar)
    # Por simplicidade, runners carregam seu próprio fluxo, mas otimizamos o load_dataset
    
    if "vit" in models:
        if verbose: print(f"\n[ViT] Inicializando pipeline...")
        model, cfg, device = load_vit_model()
        transform, _, _, _ = build_transform_from_convnext()
        
        runner = XAIPipelineRunner(
            model=model,
            transform=transform,
            device=device,
            model_type="vit",
            xai_methods=tuple(VIT_XAI_METHODS),
            xai_function=run_xai_on_image
        )
        runners["vit"] = runner
        
        if verbose: print(f"[ViT] Processando Pass 1 ({n_samples} imagens)...")
        # Carrega dados
        df = load_dataset(n_samples=n_samples)
        if not df.empty:
            res = runner.run(df, save_heatmaps=pass1_save_heatmaps, save_metrics=True, verbose=verbose)
            all_results.append(res)

    if "cnn" in models:
        if verbose: print(f"\n[CNN] Inicializando pipeline...")
        model, _, device = load_cnn_model()
        data_config = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.875} # Hardcoded simplificado ou carregar corretamente
        # Na verdade, load_cnn_model retorna data_config. O ideal seria pegar de lá.
        # Vamos assumir que load_cnn_model cuida disso.
        # Pequena correção: precisamos instanciar load_cnn_model DE NOVO ou guardar.
        # Melhor: re-instanciar ou refatorar load_cnn_model para ser cached.
        # Como o runner já instancia, vamos usar o runner.
        
        # Recarregar para garantir (os modelos são pesados, cuidado com memória)
        # Se tiver memória suficiente, ok.
        
        transform, _, _, _ = build_cnn_transform(data_config) # data_config fictício aqui, mas o runner recebe os args reais se passarmos
        
        # Nota: O código original de cnn.py retorna (model, data_config, device).
        # Vamos chamar a função corretamente.
        model, data_config, device = load_cnn_model()
        transform, _, _, _ = build_cnn_transform(data_config)

        runner = XAIPipelineRunner(
            model=model,
            transform=transform,
            device=device,
            model_type="cnn",
            xai_methods=tuple(CNN_XAI_METHODS),
            xai_function=run_xai_on_image_cnn # Importante: usar função específica da CNN
        )
        runners["cnn"] = runner
        
        if verbose: print(f"[CNN] Processando Pass 1 ({n_samples} imagens)...")
        df = load_dataset(n_samples=n_samples)
        if not df.empty:
            res = runner.run(df, save_heatmaps=pass1_save_heatmaps, save_metrics=True, verbose=verbose)
            all_results.append(res)

    # Consolida Pass 1
    combined_df = pd.concat([df for df in all_results if not df.empty], ignore_index=True)
    
    # ==========================================================================
    # 3. Execução do Pass 2 (Seleção e Visualização)
    # ==========================================================================
    
    if not combined_df.empty and HEATMAP_SELECTION_STRATEGY == "stratified":
        from stratified_selector import StratifiedSelector
        
        if verbose:
            print("\n" + "=" * 60)
            print("PASS 2: SELEÇÃO ESTRATIFICADA")
            print("=" * 60)
            
        selector = StratifiedSelector(combined_df)
        selected_df = selector.select_candidates()
        
        if not selected_df.empty:
            # Salva lista
            selection_csv = os.path.join(RESULTS_DIR, "heatmap_selection.csv")
            selected_df[["image_idx", "filename", "label", "model", "conf", "correct", "selection_reason"]].to_csv(selection_csv, index=False)
            if verbose: print(f"Lista salva em: {selection_csv}")
            
            # Gera lista de caminhos únicos e filtra DataFrame original
            # Para re-rodar, precisamos passar o subset para o runner
            # O runner.run aceita um DataFrame df.
            # Podemos criar um df filtrado.
            
            target_ids = selected_df["image_idx"].unique()
            # Precisamos do 'path', 'label', etc. que estão no df original (load_dataset)
            # Mas wait, load_dataset carrega do disco. Melhor filtrar o df retornado pelo load_dataset? 
            # Não, load_dataset pode ser chamado novamente com target_paths.
            
            target_paths = selected_df["path"].unique().tolist()
            if verbose: print(f"Gerando heatmaps para {len(target_paths)} imagens selecionadas...")
            
            # Carrega subset
            df_subset = load_dataset(target_paths=target_paths)
            
            # Roda para cada modelo ativo
            for model_name, runner in runners.items():
                if verbose: print(f"[{model_name.upper()}] Gerando heatmaps Pass 2...")
                runner.run(df_subset, save_heatmaps=True, save_metrics=False, verbose=verbose)
            
            # ==========================================================================
            # Pass 2.5: LIME/SHAP nas imagens selecionadas (opcional)
            # ==========================================================================
            if run_agnostic and AGNOSTIC_XAI_METHODS:
                if verbose:
                    print(f"\n[AGNOSTIC] Executando LIME/SHAP em {len(df_subset)} imagens...")
                
                from agnostic import run_agnostic_xai
                from visualization import save_xai_visualization
                
                # Diretório para heatmaps agnósticos
                agnostic_dir = os.path.join(HEATMAPS_DIR, "agnostic")
                os.makedirs(agnostic_dir, exist_ok=True)
                
                # Usa o primeiro modelo disponível para LIME/SHAP
                if "vit" in runners:
                    agnostic_model = runners["vit"].model
                    agnostic_device = runners["vit"].device
                    agnostic_model_type = "vit"
                elif "cnn" in runners:
                    agnostic_model = runners["cnn"].model
                    agnostic_device = runners["cnn"].device
                    agnostic_model_type = "cnn"
                else:
                    agnostic_model = None
                
                if agnostic_model is not None:
                    for idx, row in df_subset.iterrows():
                        img_path = row['path']
                        filename = row['filename']
                        true_label = row['label']
                        
                        if verbose:
                            print(f"  LIME/SHAP: {filename}...")
                        
                        try:
                            pil_img, pred_idx, conf, agnostic_maps = run_agnostic_xai(
                                img_path, agnostic_model, agnostic_device,
                                model_type=agnostic_model_type,
                                methods=tuple(AGNOSTIC_XAI_METHODS)
                            )
                            
                            # Salva visualização
                            from utils import get_label_name
                            pred_label = get_label_name(pred_idx)
                            status_str = "OK" if pred_label == true_label else "ERR"
                            heatmap_filename = f"{idx:03d}_{status_str}_{true_label}_agnostic.png"
                            heatmap_path = os.path.join(agnostic_dir, heatmap_filename)
                            title = f"AGNOSTIC | {status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                            save_xai_visualization(pil_img, agnostic_maps, heatmap_path, title, show=False)
                            
                        except Exception as e:
                            if verbose:
                                print(f"    ERRO LIME/SHAP: {e}")

    # ==========================================================================
    # 4. Finalização (Plots e Reports)
    # ==========================================================================
    
    if not combined_df.empty:
        # Salva CSV principal
        combined_df.to_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"), index=False)
        
        if generate_plots:
            if verbose: print("\n[PLOTS] Gerando gráficos...")
            generate_all_summary_plots(combined_df, SUMMARY_DIR, show=False)
            
        if generate_analysis:
            generate_analysis_report(combined_df, verbose=verbose)

    # Resumo
    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print(f"PIPELINE CONCLUÍDO em {elapsed:.1f}s")
        print("=" * 60)

    return combined_df

def main():
    parser = argparse.ArgumentParser(description="XAI Analysis Pipeline")
    parser.add_argument("--n_samples", type=int, default=None, help="Número de imagens")
    parser.add_argument("--models", nargs="+", default=["vit", "cnn"], help="Modelos: vit, cnn")
    parser.add_argument("--no-heatmaps", action="store_true", help="Não salvar heatmaps")
    parser.add_argument("--agnostic", action="store_true", help="Executar LIME/SHAP nas imagens selecionadas")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    run_full_analysis(
        n_samples=args.n_samples,
        models=args.models,
        run_agnostic=args.agnostic,
        save_heatmaps=not args.no_heatmaps,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()
