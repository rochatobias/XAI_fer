# XAI Emotion Analysis Project

Este projeto implementa um pipeline completo de Explainable AI (XAI) para análise de modelos de classificação de emoções (ViT e CNN), com foco em comparar a fidelidade e localização de diferentes métodos de atribuição.

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- PyTorch, Torchvision, TIMM
- Matplotlib, Pandas, NumPy

### Instalação
```bash
pip install -r requirements.txt
```

### Execução Rápida (Teste)
Para verificar se tudo está funcionando (processa 1 imagem):
```bash
python scripts/main.py --n_samples 1
```

### Execução Completa (Pesquisa)
Para rodar análise robusta, recomenda-se usar N=100 ou mais. O sistema usará a **Estratégia Two-Pass** para selecionar apenas os heatmaps mais relevantes (evitando salvar milhares de imagens).

```bash
python scripts/main.py --n_samples 1000 --models vit cnn
```

## 🧠 Arquitetura e Estratégias

### Estratégia "Two-Pass" (Otimização)
Para evitar o custo de I/O de salvar milhares de heatmaps inúteis:
1.  **Passo 1**: Todo o dataset é processado para calcular métricas (AOPC, Confiança, Gini) e predições. Nenhuma imagem é salva.
2.  **Seleção**: Um seletor estratificado (`stratified_selector.py`) escolhe ~140 casos representativos cobrindo 7 classes × 4 cenários (Alta Confiança Correta/Errada, Baixa Confiança Correta/Errada) com base nos percentis P80/P20.
3.  **Passo 2**: Apenas as imagens selecionadas são re-processadas para gerar e salvar as visualizações finais.

### Métricas Calculadas
O pipeline calcula automaticamente:
- **AOPC (Average Drop of Probability)**: Mede a fidelidade (quanto a remoção da área altera a predição).
- **Insertion/Deletion AUC**: Mede a qualidade do ordenamento de importância dos pixels.
- **Gini & Entropy**: Medem a dispersão/foco do heatmap.
- **MPL Curve**: Curva de Proporção de Massa vs Área (localidade).

## 📂 Estrutura de Pastas

- `scripts/`: Código fonte.
    - `main.py`: Ponto de entrada.
    - `pipeline_runner.py`: Classe que gerencia a execução dos modelos.
    - `metrics.py`: Implementação das métricas de XAI.
    - `visualization.py`: Geração de plots e heatmaps (Turbo colormap).
- `results/`: Saída do pipeline.
    - `metrics_combined.csv`: Todas as métricas para todas as imagens.
    - `heatmap_selection.csv`: Lista das imagens escolhidas para visualização.
    - `heatmaps/`: As imagens geradas (ViT e CNN).
    - `summary/`: Gráficos consolidados (Barras, Radar, Curvas).
