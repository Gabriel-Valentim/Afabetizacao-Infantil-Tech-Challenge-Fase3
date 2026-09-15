"""
Analise SHAP para interpretabilidade de modelos.

Este modulo implementa analise SHAP para explicar
as predicoes dos modelos de machine learning.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from src.utils.helpers import timer
from config import FIGURES_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Classe para analise SHAP de modelos.
    
    Gera explicacoes globais e locais usando SHAP values,
    permitindo entender como as features impactam as predicoes.
    
    Attributes:
        output_dir: Diretorio para salvar figuras
        sample_size: Tamanho da amostra para analise
    """
    
    def __init__(
        self,
        output_dir: Path = FIGURES_DIR,
        sample_size: int = 1000,
        random_state: int = RANDOM_STATE
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.random_state = random_state
        
        self._explainer: Optional[Any] = None
        self._shap_values: Optional[np.ndarray] = None
        self._X_sample: Optional[pd.DataFrame] = None
    
    @property
    def shap_values(self) -> Optional[np.ndarray]:
        """Retorna os SHAP values calculados."""
        return self._shap_values
    
    @property
    def X_sample(self) -> Optional[pd.DataFrame]:
        """Retorna a amostra utilizada."""
        return self._X_sample
    
    def _preparar_amostra(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara amostra para analise SHAP.
        
        Args:
            X: DataFrame completo
            
        Returns:
            Amostra do DataFrame
        """
        if len(X) <= self.sample_size:
            return X.copy()
        
        return X.sample(n=self.sample_size, random_state=self.random_state)
    
    @timer
    def calcular_shap_values(
        self,
        modelo: Any,
        X: pd.DataFrame,
        tipo_modelo: str = 'tree'
    ) -> np.ndarray:
        """
        Calcula SHAP values para um modelo.
        
        Args:
            modelo: Modelo treinado
            X: Features para explicacao
            tipo_modelo: Tipo do modelo ('tree' ou 'linear')
            
        Returns:
            Array com SHAP values
        """
        logger.info(f"Calculando SHAP values (amostra: {self.sample_size})...")
        
        self._X_sample = self._preparar_amostra(X)
        
        if tipo_modelo == 'tree':
            self._explainer = shap.TreeExplainer(modelo)
        elif tipo_modelo == 'linear':
            self._explainer = shap.LinearExplainer(modelo, self._X_sample)
        else:
            self._explainer = shap.Explainer(modelo, self._X_sample)
        
        self._shap_values = self._explainer.shap_values(self._X_sample)
        
        if isinstance(self._shap_values, list) and len(self._shap_values) == 2:
            self._shap_values = self._shap_values[1]
        
        logger.info(f"SHAP values calculados para {len(self._X_sample)} amostras")
        
        return self._shap_values
    
    def plot_summary(
        self,
        plot_type: str = 'dot',
        max_display: int = 20,
        salvar: bool = True
    ) -> plt.Figure:
        """
        Plota summary plot do SHAP.
        
        Args:
            plot_type: Tipo do plot ('dot', 'bar', 'violin')
            max_display: Numero maximo de features
            salvar: Se True, salva a figura
            
        Returns:
            Figura Matplotlib
        """
        if self._shap_values is None:
            raise ValueError("SHAP values nao calculados. Execute calcular_shap_values() primeiro.")
        
        logger.info(f"Gerando SHAP summary plot ({plot_type})...")
        
        plt.figure(figsize=(12, 10))
        
        shap.summary_plot(
            self._shap_values,
            self._X_sample,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.title(f'SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if salvar:
            nome_arquivo = f"shap_summary_{plot_type}.png"
            plt.savefig(
                self.output_dir / nome_arquivo,
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            logger.info(f"Salvo: {self.output_dir / nome_arquivo}")
        
        fig = plt.gcf()
        return fig
    
    def plot_bar(
        self,
        max_display: int = 20,
        salvar: bool = True
    ) -> plt.Figure:
        """
        Plota bar plot de importancia media.
        
        Args:
            max_display: Numero maximo de features
            salvar: Se True, salva a figura
            
        Returns:
            Figura Matplotlib
        """
        if self._shap_values is None:
            raise ValueError("SHAP values nao calculados.")
        
        logger.info("Gerando SHAP bar plot...")
        
        plt.figure(figsize=(12, 10))
        
        shap.summary_plot(
            self._shap_values,
            self._X_sample,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if salvar:
            plt.savefig(
                self.output_dir / "shap_bar_importance.png",
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            logger.info(f"Salvo: {self.output_dir / 'shap_bar_importance.png'}")
        
        fig = plt.gcf()
        return fig
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        salvar: bool = True
    ) -> plt.Figure:
        """
        Plota dependence plot para uma feature.
        
        Args:
            feature: Nome da feature principal
            interaction_feature: Feature para interacao (auto se None)
            salvar: Se True, salva a figura
            
        Returns:
            Figura Matplotlib
        """
        if self._shap_values is None:
            raise ValueError("SHAP values nao calculados.")
        
        if feature not in self._X_sample.columns:
            raise ValueError(f"Feature '{feature}' nao encontrada")
        
        logger.info(f"Gerando SHAP dependence plot para '{feature}'...")
        
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature,
            self._shap_values,
            self._X_sample,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence Plot - {feature}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if salvar:
            nome_arquivo = f"shap_dependence_{feature.replace(' ', '_')}.png"
            plt.savefig(
                self.output_dir / nome_arquivo,
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            logger.info(f"Salvo: {self.output_dir / nome_arquivo}")
        
        fig = plt.gcf()
        return fig
    
    def plot_waterfall(
        self,
        idx: int = 0,
        max_display: int = 15,
        salvar: bool = True
    ) -> plt.Figure:
        """
        Plota waterfall plot para uma predicao especifica.
        
        Args:
            idx: Indice da amostra
            max_display: Numero maximo de features
            salvar: Se True, salva a figura
            
        Returns:
            Figura Matplotlib
        """
        if self._shap_values is None or self._explainer is None:
            raise ValueError("SHAP values nao calculados.")
        
        logger.info(f"Gerando SHAP waterfall plot (amostra {idx})...")
        
        plt.figure(figsize=(12, 8))
        
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = self._shap_values.mean()
        
        explanation = shap.Explanation(
            values=self._shap_values[idx],
            base_values=base_value,
            data=self._X_sample.iloc[idx].values,
            feature_names=self._X_sample.columns.tolist()
        )
        
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        
        plt.title(f'SHAP Waterfall - Amostra {idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if salvar:
            plt.savefig(
                self.output_dir / f"shap_waterfall_{idx}.png",
                dpi=150, bbox_inches='tight', facecolor='white'
            )
            logger.info(f"Salvo: {self.output_dir / f'shap_waterfall_{idx}.png'}")
        
        fig = plt.gcf()
        return fig
    
    def plot_force(
        self,
        idx: int = 0,
        salvar: bool = True
    ):
        """
        Gera force plot para uma predicao especifica.
        
        Args:
            idx: Indice da amostra
            salvar: Se True, salva como HTML
            
        Returns:
            Force plot (visualizacao interativa)
        """
        if self._shap_values is None or self._explainer is None:
            raise ValueError("SHAP values nao calculados.")
        
        logger.info(f"Gerando SHAP force plot (amostra {idx})...")
        
        if hasattr(self._explainer, 'expected_value'):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = self._shap_values.mean()
        
        force_plot = shap.force_plot(
            base_value,
            self._shap_values[idx],
            self._X_sample.iloc[idx],
            matplotlib=False
        )
        
        if salvar:
            shap.save_html(
                str(self.output_dir / f"shap_force_{idx}.html"),
                force_plot
            )
            logger.info(f"Salvo: {self.output_dir / f'shap_force_{idx}.html'}")
        
        return force_plot
    
    def obter_feature_importance_shap(self) -> pd.DataFrame:
        """
        Obtem importancia das features baseada em SHAP.
        
        Returns:
            DataFrame com importancias ordenadas
        """
        if self._shap_values is None:
            raise ValueError("SHAP values nao calculados.")
        
        importancias = np.abs(self._shap_values).mean(axis=0)
        
        df_importance = pd.DataFrame({
            'feature': self._X_sample.columns,
            'shap_importance': importancias
        }).sort_values('shap_importance', ascending=False)
        
        return df_importance
    
    @timer
    def gerar_relatorio_completo(
        self,
        modelo: Any,
        X: pd.DataFrame,
        tipo_modelo: str = 'tree',
        top_features: int = 5
    ) -> Dict[str, Any]:
        """
        Gera relatorio completo de analise SHAP.
        
        Args:
            modelo: Modelo treinado
            X: Features
            tipo_modelo: Tipo do modelo
            top_features: Numero de top features para dependence plots
            
        Returns:
            Dicionario com resultados
        """
        logger.info("Gerando relatorio completo de SHAP...")
        
        self.calcular_shap_values(modelo, X, tipo_modelo)
        
        figuras = {}
        
        figuras['summary_dot'] = self.plot_summary('dot')
        plt.close()
        
        figuras['summary_bar'] = self.plot_bar()
        plt.close()
        
        df_importance = self.obter_feature_importance_shap()
        
        top_feature_names = df_importance.head(top_features)['feature'].tolist()
        for feature in top_feature_names:
            try:
                figuras[f'dependence_{feature}'] = self.plot_dependence(feature)
                plt.close()
            except Exception as e:
                logger.warning(f"Erro ao gerar dependence plot para '{feature}': {e}")
        
        for idx in [0, len(self._X_sample) // 2]:
            try:
                figuras[f'waterfall_{idx}'] = self.plot_waterfall(idx)
                plt.close()
            except Exception as e:
                logger.warning(f"Erro ao gerar waterfall para amostra {idx}: {e}")
        
        df_importance.to_csv(self.output_dir / "shap_feature_importance.csv", index=False)
        logger.info(f"Salvo: {self.output_dir / 'shap_feature_importance.csv'}")
        
        relatorio = {
            'shap_values': self._shap_values,
            'feature_importance': df_importance,
            'figuras': figuras
        }
        
        logger.info("Relatorio SHAP completo gerado!")
        
        return relatorio
