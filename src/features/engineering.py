"""
Feature Engineering - Preparacao de dados para modelagem.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.utils.helpers import timer, resumo_dataframe
from config import FEATURE_CONFIG, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe para engenharia de features."""
    
    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        feature_config=FEATURE_CONFIG,
        random_state: int = RANDOM_STATE,
        test_size: float = TEST_SIZE
    ):
        self.target_column = target_column
        self.feature_config = feature_config
        self.random_state = random_state
        self.test_size = test_size
        
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        self._feature_names: Optional[List[str]] = None
        self._numeric_features: Optional[List[str]] = None
        self._categorical_features: Optional[List[str]] = None
        self._is_fitted: bool = False
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        return self._feature_names
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def _identificar_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identifica features numericas e categoricas."""
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numericas, categoricas
    
    def _selecionar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleciona features relevantes removendo colunas excluidas."""
        colunas_excluir = self.feature_config.colunas_excluir
        colunas_remover = [c for c in colunas_excluir if c in df.columns]
        
        if colunas_remover:
            logger.info(f"Removendo {len(colunas_remover)} colunas da modelagem")
        
        df_features = df.drop(columns=colunas_remover, errors='ignore')
        return df_features
    
    @timer
    def preparar_features(
        self,
        df: pd.DataFrame,
        remover_alta_missing: bool = True,
        threshold_missing: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features para modelagem."""
        logger.info("Preparando features para modelagem...")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Coluna alvo '{self.target_column}' nao encontrada")
        
        y = df[self.target_column].copy()
        df_features = self._selecionar_features(df)
        
        if self.target_column in df_features.columns:
            df_features = df_features.drop(columns=[self.target_column])
        
        if remover_alta_missing:
            missing_pct = df_features.isnull().sum() / len(df_features)
            colunas_alta_missing = missing_pct[missing_pct > threshold_missing].index.tolist()
            
            if colunas_alta_missing:
                logger.info(f"Removendo {len(colunas_alta_missing)} colunas com >{threshold_missing*100:.0f}% missing")
                df_features = df_features.drop(columns=colunas_alta_missing)
        
        self._numeric_features, self._categorical_features = self._identificar_features(df_features)
        
        logger.info(f"Features selecionadas: Numericas={len(self._numeric_features)}, Categoricas={len(self._categorical_features)}, Total={len(df_features.columns)}")
        
        self._feature_names = df_features.columns.tolist()
        
        return df_features, y
    
    @timer
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Ajusta e aplica transformacoes nas features."""
        logger.info("Aplicando transformacoes (fit_transform)...")
        
        X = X.copy()
        
        if self._numeric_features is None:
            self._numeric_features, self._categorical_features = self._identificar_features(X)
        
        if self._numeric_features:
            logger.info(f"Imputando {len(self._numeric_features)} colunas numericas...")
            self.imputer = SimpleImputer(strategy='median')
            X[self._numeric_features] = self.imputer.fit_transform(X[self._numeric_features])
        
        if self._categorical_features:
            logger.info(f"Encoding de {len(self._categorical_features)} colunas categoricas...")
            for col in self._categorical_features:
                self.label_encoders[col] = LabelEncoder()
                X[col] = X[col].fillna('MISSING').astype(str)
                X[col] = self.label_encoders[col].fit_transform(X[col])
        
        if self._numeric_features:
            logger.info("Normalizando features numericas...")
            self.scaler = StandardScaler()
            X[self._numeric_features] = self.scaler.fit_transform(X[self._numeric_features])
        
        self._is_fitted = True
        logger.info("Transformacoes aplicadas!")
        
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformacoes ja ajustadas em novos dados."""
        if not self._is_fitted:
            raise ValueError("FeatureEngineer nao foi fitted. Chame fit_transform() primeiro.")
        
        X = X.copy()
        
        if self.imputer and self._numeric_features:
            X[self._numeric_features] = self.imputer.transform(X[self._numeric_features])
        
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna('MISSING').astype(str)
                X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else 'MISSING')
                X[col] = encoder.transform(X[col])
        
        if self.scaler and self._numeric_features:
            X[self._numeric_features] = self.scaler.transform(X[self._numeric_features])
        
        return X
    
    @timer
    def split_treino_teste(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide os dados em conjuntos de treino e teste."""
        logger.info(f"Dividindo dados em treino ({1-self.test_size:.0%}) e teste ({self.test_size:.0%})...")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")
        
        if stratify:
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            logger.info(f"Distribuicao treino: {train_dist[1]:.2%} positivos | Teste: {test_dist[1]:.2%} positivos")
        
        return X_train, X_test, y_train, y_test
    
    @timer
    def pipeline_completo(
        self,
        df: pd.DataFrame,
        remover_alta_missing: bool = True,
        threshold_missing: float = 0.5,
        normalizar: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Executa o pipeline completo de feature engineering."""
        logger.info("Executando pipeline de Feature Engineering...")
        
        X, y = self.preparar_features(df, remover_alta_missing, threshold_missing)
        X_train, X_test, y_train, y_test = self.split_treino_teste(X, y)
        X_train = self.fit_transform(X_train)
        X_test = self.transform(X_test)
        
        logger.info(f"Pipeline de Feature Engineering finalizado! Features finais: {len(self._feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """Retorna os nomes das features apos processamento."""
        if self._feature_names is None:
            raise ValueError("Features nao processadas. Execute pipeline_completo() primeiro.")
        return self._feature_names
