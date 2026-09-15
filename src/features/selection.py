"""
Feature Selection - Selecao de features relevantes.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFE

from src.utils.helpers import timer
from config import RANDOM_STATE

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Classe para selecao de features."""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.selected_features: Optional[List[str]] = None
        self._importance_scores: Optional[pd.DataFrame] = None
    
    @property
    def importance_scores(self) -> Optional[pd.DataFrame]:
        return self._importance_scores
    
    @timer
    def calcular_importancia_rf(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 10
    ) -> pd.DataFrame:
        """Calcula importancia das features usando Random Forest."""
        logger.info("Calculando importancia via Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        importancia = pd.DataFrame({
            'feature': X.columns,
            'importance_rf': rf.feature_importances_
        }).sort_values('importance_rf', ascending=False)
        
        logger.info(f"Top 10 features por RF:")
        for _, row in importancia.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance_rf']:.4f}")
        
        return importancia
    
    @timer
    def calcular_correlacao(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula correlacao das features com o target."""
        logger.info("Calculando correlacao com target...")
        
        correlacoes = []
        for col in X.columns:
            corr = X[col].corr(y)
            correlacoes.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        
        df_corr = pd.DataFrame(correlacoes).sort_values('abs_correlation', ascending=False)
        
        logger.info(f"Top 10 features por correlacao:")
        for _, row in df_corr.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['correlation']:.4f}")
        
        return df_corr
    
    @timer
    def calcular_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calcula informacao mutua das features com o target."""
        logger.info("Calculando informacao mutua...")
        
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state, n_jobs=-1)
        
        df_mi = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        logger.info(f"Top 10 features por MI:")
        for _, row in df_mi.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['mutual_info']:.4f}")
        
        return df_mi
    
    @timer
    def calcular_scores_combinados(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Combina multiplos metodos de scoring."""
        logger.info("Calculando scores combinados...")
        
        df_rf = self.calcular_importancia_rf(X, y)
        df_corr = self.calcular_correlacao(X, y)
        df_mi = self.calcular_mutual_info(X, y)
        
        df_scores = df_rf.merge(
            df_corr[['feature', 'abs_correlation']], on='feature'
        ).merge(
            df_mi, on='feature'
        )
        
        for col in ['importance_rf', 'abs_correlation', 'mutual_info']:
            min_val = df_scores[col].min()
            max_val = df_scores[col].max()
            if max_val > min_val:
                df_scores[f'{col}_norm'] = (df_scores[col] - min_val) / (max_val - min_val)
            else:
                df_scores[f'{col}_norm'] = 0
        
        df_scores['score_combinado'] = (
            df_scores['importance_rf_norm'] +
            df_scores['abs_correlation_norm'] +
            df_scores['mutual_info_norm']
        ) / 3
        
        df_scores = df_scores.sort_values('score_combinado', ascending=False)
        self._importance_scores = df_scores
        
        logger.info("Scores combinados calculados!")
        logger.info(f"Top 10 features (score combinado):")
        for _, row in df_scores.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['score_combinado']:.4f}")
        
        return df_scores
    
    @timer
    def selecionar_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[str]:
        """Seleciona as top N features ou features acima de um threshold."""
        if self._importance_scores is None:
            self.calcular_scores_combinados(X, y)
        
        df_scores = self._importance_scores
        
        if threshold is not None:
            selected = df_scores[df_scores['score_combinado'] >= threshold]['feature'].tolist()
            logger.info(f"Selecionadas {len(selected)} features com score >= {threshold}")
        elif n_features is not None:
            selected = df_scores.head(n_features)['feature'].tolist()
            logger.info(f"Selecionadas top {n_features} features")
        else:
            n_default = max(20, len(df_scores) // 2)
            selected = df_scores.head(n_default)['feature'].tolist()
            logger.info(f"Selecionadas top {len(selected)} features (padrao)")
        
        self.selected_features = selected
        return selected
    
    @timer
    def selecionar_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """Seleciona features usando Recursive Feature Elimination."""
        logger.info(f"Selecionando features via RFE (n={n_features})...")
        
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=5)
        rfe.fit(X, y)
        
        selected = X.columns[rfe.support_].tolist()
        
        logger.info(f"{len(selected)} features selecionadas via RFE")
        
        self.selected_features = selected
        return selected
    
    def filtrar_features(self, X: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Filtra DataFrame mantendo apenas as features selecionadas."""
        if features is None:
            features = self.selected_features
        
        if features is None:
            raise ValueError("Nenhuma feature selecionada. Execute um metodo de selecao primeiro.")
        
        features_existentes = [f for f in features if f in X.columns]
        
        if len(features_existentes) < len(features):
            logger.warning(f"{len(features) - len(features_existentes)} features nao encontradas")
        
        return X[features_existentes]
