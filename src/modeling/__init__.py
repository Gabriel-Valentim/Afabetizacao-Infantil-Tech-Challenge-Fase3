"""
Módulo de Modelagem de Machine Learning.

Responsável por:
- Treinamento de modelos (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Avaliação de métricas
- Comparação de modelos
- Persistência de modelos treinados
"""

from .trainer import ModelTrainer
from .evaluator import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator']
