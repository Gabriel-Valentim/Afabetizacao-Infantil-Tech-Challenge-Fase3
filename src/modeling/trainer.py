"""
Treinamento de Modelos de Machine Learning.

Este modulo implementa o treinamento de multiplos modelos:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.helpers import timer
from config import MODEL_CONFIG, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Classe para treinamento de modelos de classificacao.
    
    Treina e gerencia multiplos modelos de ML para predicao
    de alfabetizacao.
    
    Attributes:
        model_config: Configuracoes dos modelos
        models_dir: Diretorio para salvar modelos
        models: Dicionario com modelos treinados
    """
    
    def __init__(
        self,
        model_config=MODEL_CONFIG,
        models_dir: Path = MODELS_DIR
    ):
        self.model_config = model_config
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self._is_fitted: bool = False
    
    @property
    def is_fitted(self) -> bool:
        """Retorna se os modelos foram treinados."""
        return self._is_fitted
    
    def _criar_modelos(self) -> Dict[str, Any]:
        """
        Cria instancias dos modelos com hiperparametros configurados.
        
        Returns:
            Dicionario com modelos instanciados
        """
        modelos = {
            'logistic_regression': LogisticRegression(**self.model_config.lr_params),
            'random_forest': RandomForestClassifier(**self.model_config.rf_params),
            'xgboost': XGBClassifier(**self.model_config.xgb_params),
            'lightgbm': LGBMClassifier(**self.model_config.lgbm_params)
        }
        
        return modelos
    
    @timer
    def treinar_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """
        Treina modelo de Logistic Regression.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo treinado
        """
        logger.info("MODELO 1: LOGISTIC REGRESSION (Baseline)")
        logger.info("=" * 60)
        
        modelo = LogisticRegression(**self.model_config.lr_params)
        modelo.fit(X_train, y_train)
        
        self.models['logistic_regression'] = modelo
        logger.info("Logistic Regression treinado!")
        
        return modelo
    
    @timer
    def treinar_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Treina modelo Random Forest.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo treinado
        """
        logger.info("MODELO 2: RANDOM FOREST")
        logger.info("=" * 60)
        
        modelo = RandomForestClassifier(**self.model_config.rf_params)
        modelo.fit(X_train, y_train)
        
        self.models['random_forest'] = modelo
        logger.info("Random Forest treinado!")
        
        return modelo
    
    @timer
    def treinar_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> XGBClassifier:
        """
        Treina modelo XGBoost.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo treinado
        """
        logger.info("MODELO 3: XGBOOST")
        logger.info("=" * 60)
        
        modelo = XGBClassifier(**self.model_config.xgb_params)
        modelo.fit(X_train, y_train)
        
        self.models['xgboost'] = modelo
        logger.info("XGBoost treinado!")
        
        return modelo
    
    @timer
    def treinar_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LGBMClassifier:
        """
        Treina modelo LightGBM.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Modelo treinado
        """
        logger.info("MODELO 4: LIGHTGBM")
        logger.info("=" * 60)
        
        modelo = LGBMClassifier(**self.model_config.lgbm_params)
        modelo.fit(X_train, y_train)
        
        self.models['lightgbm'] = modelo
        logger.info("LightGBM treinado!")
        
        return modelo
    
    @timer
    def treinar_todos_modelos(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Treina todos os modelos configurados.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Dicionario com modelos treinados
        """
        logger.info("=" * 60)
        logger.info("INICIANDO TREINAMENTO DE TODOS OS MODELOS")
        logger.info("=" * 60)
        
        self.treinar_logistic_regression(X_train, y_train)
        self.treinar_random_forest(X_train, y_train)
        self.treinar_xgboost(X_train, y_train)
        self.treinar_lightgbm(X_train, y_train)
        
        self._is_fitted = True
        
        logger.info("=" * 60)
        logger.info("TODOS OS MODELOS TREINADOS COM SUCESSO!")
        logger.info("=" * 60)
        
        return self.models
    
    def predizer(
        self,
        X: pd.DataFrame,
        modelo_nome: str = 'xgboost'
    ) -> np.ndarray:
        """
        Realiza predicoes com um modelo especifico.
        
        Args:
            X: Features para predicao
            modelo_nome: Nome do modelo a usar
            
        Returns:
            Array com predicoes
        """
        if modelo_nome not in self.models:
            raise ValueError(f"Modelo '{modelo_nome}' nao encontrado. Modelos disponiveis: {list(self.models.keys())}")
        
        return self.models[modelo_nome].predict(X)
    
    def predizer_proba(
        self,
        X: pd.DataFrame,
        modelo_nome: str = 'xgboost'
    ) -> np.ndarray:
        """
        Retorna probabilidades de predicao.
        
        Args:
            X: Features para predicao
            modelo_nome: Nome do modelo a usar
            
        Returns:
            Array com probabilidades (classe positiva)
        """
        if modelo_nome not in self.models:
            raise ValueError(f"Modelo '{modelo_nome}' nao encontrado")
        
        return self.models[modelo_nome].predict_proba(X)[:, 1]
    
    def predizer_todos(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Realiza predicoes com todos os modelos.
        
        Args:
            X: Features para predicao
            
        Returns:
            Dicionario com predicoes de cada modelo
        """
        predicoes = {}
        
        for nome, modelo in self.models.items():
            predicoes[nome] = {
                'y_pred': modelo.predict(X),
                'y_proba': modelo.predict_proba(X)[:, 1]
            }
        
        return predicoes
    
    def salvar_modelo(
        self,
        modelo_nome: str,
        nome_arquivo: Optional[str] = None
    ) -> Path:
        """
        Salva um modelo especifico em disco.
        
        Args:
            modelo_nome: Nome do modelo a salvar
            nome_arquivo: Nome do arquivo (opcional)
            
        Returns:
            Path do arquivo salvo
        """
        if modelo_nome not in self.models:
            raise ValueError(f"Modelo '{modelo_nome}' nao encontrado")
        
        if nome_arquivo is None:
            nome_arquivo = f"modelo_{modelo_nome}.pkl"
        
        arquivo = self.models_dir / nome_arquivo
        joblib.dump(self.models[modelo_nome], arquivo)
        
        logger.info(f"Modelo salvo: {arquivo}")
        
        return arquivo
    
    def salvar_todos_modelos(self) -> Dict[str, Path]:
        """
        Salva todos os modelos treinados.
        
        Returns:
            Dicionario com paths dos arquivos salvos
        """
        arquivos = {}
        
        for nome in self.models.keys():
            arquivos[nome] = self.salvar_modelo(nome)
        
        logger.info(f"{len(arquivos)} modelos salvos em: {self.models_dir}")
        
        return arquivos
    
    def carregar_modelo(
        self,
        modelo_nome: str,
        nome_arquivo: Optional[str] = None
    ) -> Any:
        """
        Carrega um modelo do disco.
        
        Args:
            modelo_nome: Nome do modelo
            nome_arquivo: Nome do arquivo (opcional)
            
        Returns:
            Modelo carregado
        """
        if nome_arquivo is None:
            nome_arquivo = f"modelo_{modelo_nome}.pkl"
        
        arquivo = self.models_dir / nome_arquivo
        
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo}")
        
        modelo = joblib.load(arquivo)
        self.models[modelo_nome] = modelo
        
        logger.info(f"Modelo carregado: {arquivo}")
        
        return modelo
    
    def obter_feature_importance(
        self,
        modelo_nome: str,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Obtem importancia das features de um modelo.
        
        Args:
            modelo_nome: Nome do modelo
            feature_names: Nomes das features
            
        Returns:
            DataFrame com importancias
        """
        if modelo_nome not in self.models:
            raise ValueError(f"Modelo '{modelo_nome}' nao encontrado")
        
        modelo = self.models[modelo_nome]
        
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
        elif hasattr(modelo, 'coef_'):
            importancias = np.abs(modelo.coef_[0])
        else:
            raise ValueError(f"Modelo '{modelo_nome}' nao suporta feature importance")
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importancias
        }).sort_values('importance', ascending=False)
        
        return df_importance
