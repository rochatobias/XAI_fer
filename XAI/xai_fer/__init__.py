"""
XAI para Reconhecimento de Emoções Faciais (FER).

Pacote para análise de explicabilidade (XAI) aplicada a modelos de deep learning
(Vision Transformer e ConvNeXt) na tarefa de reconhecimento de emoções faciais.

Subpacotes:
    - data: carregamento e amostragem do dataset FER.
    - models: explicadores específicos (ViT, CNN) e agnósticos (LIME, SHAP).
    - evaluation: métricas de fidelidade/localidade e geração de relatórios CSV.
    - visualization: heatmaps individuais e gráficos de sumário.
    - pipeline: orquestração (runner) e seleção estratificada de amostras.
    - reporting: geração de tabelas LaTeX para publicação.
"""

__version__ = "2.0.0"
