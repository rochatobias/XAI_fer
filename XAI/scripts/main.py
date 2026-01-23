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
                from agnostic import run_agnostic_xai
                from visualization import save_xai_visualization
                from utils import get_label_name
                
                # Determina quantas imagens processar com LIME/SHAP
                n_stratified = len(df_subset)
                n_agnostic = min(n_samples_agnostic, n_stratified)
                
                if n_samples_agnostic > n_stratified:
                    print(f"\n[AGNOSTIC] Aviso: N_SAMPLES_AGNOSTIC={n_samples_agnostic} > imagens estratificadas={n_stratified}")
                    print(f"           Serão analisadas apenas {n_stratified} imagens (limite do estratificado)")
                
                # Limita ao N_SAMPLES_AGNOSTIC
                df_agnostic = df_subset.head(n_agnostic)
                
                # Roda LIME/SHAP para CADA modelo ativo
                for model_name, runner in runners.items():
                    if verbose:
                        print(f"\n[AGNOSTIC-{model_name.upper()}] Executando LIME/SHAP em {n_agnostic} imagens...")
                    
                    # Diretório separado por modelo
                    agnostic_model_dir = os.path.join(HEATMAPS_DIR, "agnostic", model_name)
                    os.makedirs(agnostic_model_dir, exist_ok=True)
                    
                    agnostic_model = runner.model
                    agnostic_device = runner.device
                    
                    for i, (idx, row) in enumerate(df_agnostic.iterrows()):
                        img_path = row['path']
                        filename = row['filename']
                        true_label = row['label']
                        
                        if verbose:
                            print(f"  [{i+1}/{n_agnostic}] LIME/SHAP ({model_name}): {filename}...")
                        
                        try:
                            pil_img, pred_idx, conf, agnostic_maps = run_agnostic_xai(
                                img_path, agnostic_model, agnostic_device,
                                model_type=model_name,
                                methods=tuple(AGNOSTIC_XAI_METHODS)
                            )
                            
                            # Salva visualização
                            pred_label = get_label_name(pred_idx)
                            status_str = "OK" if pred_label == true_label else "ERR"
                            heatmap_filename = f"{idx:03d}_{status_str}_{true_label}_agnostic.png"
                            heatmap_path = os.path.join(agnostic_model_dir, heatmap_filename)
                            title = f"AGNOSTIC-{model_name.upper()} | {status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
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


def run_agnostic_only(
    models: list = None,
    verbose: bool = True
) -> None:
    """
    Executa apenas LIME/SHAP nas imagens já selecionadas pelo estratificado.
    Lê o CSV existente e roda para todas as imagens únicas.
    """
    from tqdm import tqdm
    
    start_time = time.time()
    create_results_dirs()
    
    if models is None:
        models = ["vit", "cnn"]
    
    # Verifica se existe CSV de seleção
    selection_csv = os.path.join(RESULTS_DIR, "heatmap_selection.csv")
    if not os.path.exists(selection_csv):
        print(f"ERRO: {selection_csv} não encontrado. Execute o pipeline completo primeiro.")
        return
    
    if verbose:
        print("\n" + "=" * 60)
        print("AGNOSTIC-ONLY MODE: LIME/SHAP")
        print("=" * 60)
    
    # Lê CSV e extrai caminhos únicos
    selection_df = pd.read_csv(selection_csv)
    unique_filenames = selection_df["filename"].unique().tolist()
    
    if verbose:
        print(f"[AGNOSTIC] {len(unique_filenames)} filenames únicos no CSV")
    
    # Carrega TODAS as imagens do diretório e filtra pelos filenames
    from data_loader import get_all_images
    all_images = get_all_images()
    
    # Filtra pelas imagens selecionadas
    filtered = [img for img in all_images if img["filename"] in unique_filenames]
    df_agnostic = pd.DataFrame(filtered)
    
    # Adiciona image_idx do CSV original para manter consistência nos nomes
    idx_map = selection_df.drop_duplicates("filename").set_index("filename")["image_idx"]
    df_agnostic["image_idx"] = df_agnostic["filename"].map(idx_map)
    
    n_images = len(df_agnostic)
    if verbose:
        print(f"[AGNOSTIC] {n_images} imagens únicas para processar")
    
    # Carrega modelos e executa para cada um
    for model_name in models:
        if verbose:
            print(f"\n[AGNOSTIC-{model_name.upper()}] Carregando modelo...")
        
        if model_name == "vit":
            model, cfg, device = load_vit_model()
        else:
            model, data_config, device = load_cnn_model()
        
        agnostic_dir = os.path.join(HEATMAPS_DIR, "agnostic", model_name)
        os.makedirs(agnostic_dir, exist_ok=True)
        
        if verbose:
            print(f"[AGNOSTIC-{model_name.upper()}] Executando LIME/SHAP em {n_images} imagens...")
        
        # Loop com tqdm para progresso visual
        for idx, row in tqdm(df_agnostic.iterrows(), total=n_images, 
                             desc=f"LIME/SHAP ({model_name.upper()})", unit="img"):
            img_path = row['path']
            filename = row['filename']
            true_label = row['label']
            image_idx = row.get('image_idx', idx)
            
            try:
                pil_img, pred_idx, conf, agnostic_maps = run_agnostic_xai(
                    img_path, model, device,
                    model_type=model_name,
                    methods=tuple(AGNOSTIC_XAI_METHODS)
                )
                
                from utils import get_label_name
                from visualization import save_xai_visualization
                
                pred_label = get_label_name(pred_idx)
                status_str = "OK" if pred_label == true_label else "ERR"
                heatmap_filename = f"{image_idx:03d}_{status_str}_{true_label}_agnostic.png"
                heatmap_path = os.path.join(agnostic_dir, heatmap_filename)
                title = f"AGNOSTIC-{model_name.upper()} | {status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                save_xai_visualization(pil_img, agnostic_maps, heatmap_path, title, show=False)
                
            except Exception as e:
                if verbose:
                    tqdm.write(f"  ERRO {filename}: {e}")
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n[AGNOSTIC] Concluído em {elapsed:.1f}s")
        print(f"[AGNOSTIC] Heatmaps salvos em: {os.path.join(HEATMAPS_DIR, 'agnostic')}")

# ==============================================================================
# FUNÇÕES AUXILIARES PARA MENU INTERATIVO
# ==============================================================================

def run_plots_only(verbose: bool = True) -> None:
    """Regenera apenas os gráficos a partir do CSV existente."""
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
        print(f"[PLOTS] Gráficos regenerados com sucesso!")


def run_analysis_only(verbose: bool = True) -> None:
    """Regenera apenas os CSVs de análise a partir do CSV existente."""
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
        print(f"[ANALYSIS] CSVs de análise regenerados!")


def show_menu() -> None:
    """Exibe menu principal."""
    print("\n" + "═" * 60)
    print("         XAI ANALYSIS - MENU PRINCIPAL")
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


def prompt_int(prompt: str, default: int = None) -> int:
    """Solicita um inteiro ao usuário."""
    while True:
        try:
            default_str = f" [{default}]" if default else ""
            user_input = input(f"{prompt}{default_str}: ").strip()
            if not user_input and default is not None:
                return default
            return int(user_input)
        except ValueError:
            print("Por favor, digite um número inteiro.")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Solicita confirmação sim/não."""
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
                # Pipeline completo ViT + CNN
                n = prompt_int("Número de imagens para análise", N_SAMPLES)
                agnostic = prompt_yes_no("Executar LIME/SHAP também?", False)
                n_agnostic = None
                if agnostic:
                    n_agnostic = prompt_int("Número de imagens para LIME/SHAP", N_SAMPLES_AGNOSTIC)
                
                run_full_analysis(
                    n_samples=n,
                    n_samples_agnostic=n_agnostic,
                    models=["vit", "cnn"],
                    run_agnostic=agnostic,
                    verbose=True
                )
                
            elif choice == "2":
                # Apenas ViT
                n = prompt_int("Número de imagens para análise", N_SAMPLES)
                run_full_analysis(n_samples=n, models=["vit"], verbose=True)
                
            elif choice == "3":
                # Apenas CNN
                n = prompt_int("Número de imagens para análise", N_SAMPLES)
                run_full_analysis(n_samples=n, models=["cnn"], verbose=True)
                
            elif choice == "4":
                # LIME/SHAP apenas
                models_input = input("Modelos (vit, cnn ou ambos) [vit cnn]: ").strip()
                if not models_input:
                    models = ["vit", "cnn"]
                else:
                    models = models_input.split()
                run_agnostic_only(models=models, verbose=True)
                
            elif choice == "5":
                # Regenerar plots
                run_plots_only(verbose=True)
                
            elif choice == "6":
                # Regenerar análise
                run_analysis_only(verbose=True)
                
            else:
                print("Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n\nOperação cancelada.")
            break
        except Exception as e:
            print(f"\nErro: {e}")


def main():
    """Função principal com suporte a CLI e menu interativo."""
    
    # Se não há argumentos, mostra menu interativo
    if len(sys.argv) == 1:
        run_interactive_menu()
        return
    
    # Parse de argumentos CLI
    parser = argparse.ArgumentParser(
        description="XAI Analysis Pipeline para Reconhecimento de Emoções",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                         # Menu interativo
  python main.py --n_samples 100         # Pipeline completo com 100 imagens
  python main.py --models vit            # Apenas ViT
  python main.py --agnostic-only         # Apenas LIME/SHAP (usa CSV existente)
  python main.py --plots-only            # Regenerar gráficos
  python main.py --n_samples 50 --agnostic --n_agnostic 10  # Pipeline + LIME/SHAP
        """
    )
    
    # Argumentos de quantidade
    parser.add_argument("--n_samples", "-n", type=int, default=None,
                        help=f"Número de imagens para análise (default: {N_SAMPLES})")
    parser.add_argument("--n_agnostic", type=int, default=None,
                        help=f"Número de imagens para LIME/SHAP (default: {N_SAMPLES_AGNOSTIC})")
    
    # Argumentos de seleção de modelo
    parser.add_argument("--models", "-m", nargs="+", default=["vit", "cnn"],
                        choices=["vit", "cnn"],
                        help="Modelos a processar (default: vit cnn)")
    
    # Flags de execução
    parser.add_argument("--agnostic", "-a", action="store_true",
                        help="Executar LIME/SHAP nas imagens estratificadas")
    parser.add_argument("--agnostic-only", action="store_true",
                        help="Executar APENAS LIME/SHAP (usa CSV existente)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerar apenas gráficos (usa CSV existente)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Regenerar apenas CSVs de análise")
    
    # Flags de controle
    parser.add_argument("--no-heatmaps", action="store_true",
                        help="Não salvar heatmaps visuais")
    parser.add_argument("--no-plots", action="store_true",
                        help="Não gerar gráficos de resumo")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Não gerar CSVs de análise")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Modo silencioso")
    
    args = parser.parse_args()
    
    # Executa a ação apropriada
    if args.plots_only:
        run_plots_only(verbose=not args.quiet)
        
    elif args.analysis_only:
        run_analysis_only(verbose=not args.quiet)
        
    elif args.agnostic_only:
        run_agnostic_only(
            models=args.models,
            verbose=not args.quiet
        )
        
    else:
        run_full_analysis(
            n_samples=args.n_samples,
            n_samples_agnostic=args.n_agnostic,
            models=args.models,
            run_agnostic=args.agnostic,
            save_heatmaps=not args.no_heatmaps,
            generate_plots=not args.no_plots,
            generate_analysis=not args.no_analysis,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
