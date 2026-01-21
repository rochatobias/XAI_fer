# ==============================================================================
# Main - Pipeline Principal XAI
# ==============================================================================
# Execute este arquivo para rodar toda a análise XAI automaticamente
# Modifique N_SAMPLES em config.py para controlar quantas imagens processar
# ==============================================================================

import os
import sys
import time
import pandas as pd
import numpy as np

from config import (
    N_SAMPLES, MODEL_DIR, RESULTS_DIR, HEATMAPS_DIR, SUMMARY_DIR,
    XAI_METHODS, MEAN, STD, create_results_dirs, print_config, get_device
)
from data_loader import load_dataset
from vit import load_vit_model, build_transform_from_convnext, run_xai_on_image
from metrics import compute_all_metrics
from visualization import save_xai_visualization, generate_all_summary_plots
from utils import get_label_name


def run_xai_pipeline(n_samples: int = None, save_heatmaps: bool = True, save_metrics: bool = True, generate_plots: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Pipeline principal que executa toda a análise XAI."""
    start_time = time.time()
    if n_samples is None:
        n_samples = N_SAMPLES

    # 1. Setup
    if verbose:
        print("\n" + "=" * 60)
        print("XAI ANALYSIS PIPELINE")
        print("=" * 60)
        print_config()
    create_results_dirs()

    # 2. Carregar modelo
    if verbose:
        print("\n[1/5] Carregando modelo ViT...")
    model, cfg, device = load_vit_model()
    transform, img_size, mean, std = build_transform_from_convnext()

    # 3. Carregar dataset
    if verbose:
        print(f"\n[2/5] Carregando dataset ({n_samples} imagens)...")
    df = load_dataset(n_samples=n_samples)
    if len(df) == 0:
        print("ERRO: Nenhuma imagem encontrada!")
        return pd.DataFrame()

    # 4. Processar cada imagem
    if verbose:
        print(f"\n[3/5] Processando {len(df)} imagens com XAI...")
    all_results = []
    for idx, row in df.iterrows():
        img_path = row['path']
        true_label = row['label']
        true_label_idx = row['label_idx']
        filename = row['filename']
        if verbose:
            print(f"\n  [{idx+1}/{len(df)}] {filename}")
            print(f"       Label: {true_label}")
        try:
            pil_img, pred_idx, conf, maps = run_xai_on_image(img_path, model, transform, device, methods=tuple(XAI_METHODS))
            pred_label = get_label_name(pred_idx)
            correct = (pred_idx == true_label_idx)
            if verbose:
                status = "✓" if correct else "✗"
                print(f"       Pred: {pred_label} ({conf:.2%}) {status}")
            
            # Salvar visualização
            if save_heatmaps:
                status_str = "OK" if correct else "ERR"
                heatmap_path = os.path.join(HEATMAPS_DIR, f"{idx:03d}_{status_str}_{true_label}_pred{pred_label}_{conf:.2f}.png")
                title = f"{status_str} | True={true_label} | Pred={pred_label} | Conf={conf:.2%}"
                save_xai_visualization(pil_img, maps, heatmap_path, title, show=False)
            
            # Calcular métricas para cada método XAI
            for method_name, heatmap in maps.items():
                if verbose:
                    print(f"       Calculando métricas: {method_name}...")
                metrics = compute_all_metrics(model, pil_img, heatmap, device, mean, std, true_label_idx=true_label_idx)
                result = {
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
                    'AOPC': metrics['AOPC'],
                    'Insertion_AUC': metrics['Insertion_AUC'],
                    'Deletion_AUC': metrics['Deletion_AUC'],
                    'Area@50': metrics['Area@50'],
                    'Area@90': metrics['Area@90'],
                    'MPL_AUC': metrics['MPL_AUC'],
                    'Entropy': metrics['Entropy'],
                    'Gini': metrics['Gini'],
                }
                all_results.append(result)
        except Exception as e:
            print(f"       ERRO processando {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 5. Criar DataFrame de resultados
    if verbose:
        print(f"\n[4/5] Compilando resultados...")
    results_df = pd.DataFrame(all_results)
    if len(results_df) == 0:
        print("ERRO: Nenhum resultado foi gerado!")
        return results_df

    # Salvar CSV
    if save_metrics:
        csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"       Métricas salvas em: {csv_path}")

    # 6. Gerar gráficos de sumário
    if generate_plots and len(results_df) > 0:
        if verbose:
            print(f"\n[5/5] Gerando gráficos de sumário...")
        generate_all_summary_plots(results_df, SUMMARY_DIR, show=False)

    # 7. Resumo final
    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 60)
        print("RESUMO DA ANÁLISE")
        print("=" * 60)
        print(f"Imagens processadas: {len(df)}")
        print(f"Métodos XAI: {', '.join(XAI_METHODS)}")
        print(f"Total de registros: {len(results_df)}")
        print(f"Tempo total: {elapsed:.1f}s ({elapsed/len(df):.1f}s/imagem)")
        print(f"\nAcurácia do modelo nas amostras:")
        accuracy = results_df.groupby('image_idx')['correct'].first().mean()
        print(f"  {accuracy:.1%} ({results_df.groupby('image_idx')['correct'].first().sum()}/{len(df)})")
        print(f"\nMétricas médias por método:")
        summary = results_df.groupby('method')[['AOPC', 'Insertion_AUC', 'Deletion_AUC']].mean()
        print(summary.to_string())
        print(f"\nResultados salvos em: {RESULTS_DIR}")
        print("=" * 60)
    return results_df


def main():
    """Função principal - ponto de entrada."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  XAI para Classificação de Emoções (ViT)".center(56) + "  #")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    results = run_xai_pipeline(
        n_samples=None,
        save_heatmaps=True,
        save_metrics=True,
        generate_plots=True,
        verbose=True
    )
    
    if len(results) > 0:
        print("\n✓ Pipeline concluído com sucesso!")
        print(f"\nPara alterar o número de imagens, edite N_SAMPLES em:")
        print(f"  {os.path.join(os.path.dirname(__file__), 'config.py')}")
    else:
        print("\n✗ Pipeline falhou - verifique os erros acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
