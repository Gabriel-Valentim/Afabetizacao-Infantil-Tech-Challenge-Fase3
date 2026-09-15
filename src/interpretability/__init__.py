"""
Módulo de Interpretabilidade de Modelos.

Responsável por:
- Análise SHAP (SHapley Additive exPlanations)
- Visualizações de explicabilidade
- Feature importance global e local
"""

from .shap_analysis import SHAPAnalyzer

__all__ = ['SHAPAnalyzer']
