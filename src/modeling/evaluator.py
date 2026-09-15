"""
Avaliacao de Modelos de Machine Learning.

Este modulo implementa avaliacao completa de modelos:
- Metricas de classificacao
- Curvas ROC e Precision-Recall
- Matrizes de confusao
- Comparacao entre modelos
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

from src.utils.helpers import timer
from config import VIZ_CONFIG, FIGURES_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Classe para avaliacao de modelos de classificacao.
    
    Calcula metricas, gera visualizacoes e compara
    performance entre modelos.
    
    Attributes:
        viz_config: Configuracoes de visualizacao
        output_dir: Diretorio para salvar figuras
    """
    
    def __init__(
        self,
        viz_config=VIZ_CONFIG,
        output_dir: Path = FIGURES_DIR
    ):
        self.viz_config = viz_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calcular_metricas(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcula metricas de classificacao.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            y_proba: Probabilidades (opcional)
            
        Returns:
            Dicionario com metricas
        """
        metricas = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0)
        }
        
        if y_proba is not None:
            metricas['roc_auc'] = roc_auc_score(y_true, y_proba)
            metricas['avg_precision'] = average_precision_score(y_true, y_proba)
        
        return metricas
    
    @timer
    def avaliar_modelo(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        nome_modelo: str = "Modelo"
    ) -> Dict[str, float]:
        """
        Avalia um modelo e exibe metricas.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            y_proba: Probabilidades (opcional)
            nome_modelo: Nome do modelo
            
        Returns:
            Dicionario com metricas
        """
        metricas = self.calcular_metricas(y_true, y_pred, y_proba)
        
        logger.info(f"""
{'='*60}
AVALIACAO: {nome_modelo}
{'='*60}
Acuracia:    {metricas['accuracy']:.4f}
Precisao:    {metricas['precision']:.4f}
Recall:      {metricas['recall']:.4f}
F1-Score:    {metricas['f1_score']:.4f}
Especificidade: {metricas['specificity']:.4f}
""")
        
        if 'roc_auc' in metricas:
            logger.info(f"ROC-AUC:     {metricas['roc_auc']:.4f}")
            logger.info(f"Avg Precision: {metricas['avg_precision']:.4f}")
        
        return metricas
    
    @timer
    def avaliar_todos_modelos(
        self,
        y_true: np.ndarray,
        predicoes: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Avalia todos os modelos e gera comparativo.
        
        Args:
            y_true: Valores reais
            predicoes: Dicionario {modelo: {y_pred, y_proba}}
            
        Returns:
            DataFrame com metricas comparativas
        """
        resultados = []
        
        for nome, preds in predicoes.items():
            metricas = self.calcular_metricas(
                y_true,
                preds['y_pred'],
                preds.get('y_proba')
            )
            metricas['modelo'] = nome
            resultados.append(metricas)
        
        df_resultados = pd.DataFrame(resultados)
        
        cols_ordem = ['modelo', 'accuracy', 'precision', 'recall', 'f1_score', 
                      'specificity', 'roc_auc', 'avg_precision']
        cols_disponiveis = [c for c in cols_ordem if c in df_resultados.columns]
        df_resultados = df_resultados[cols_disponiveis]
        
        df_resultados = df_resultados.sort_values('f1_score', ascending=False)
        
        logger.info(f"""
{'='*70}
COMPARATIVO DE MODELOS
{'='*70}
""")
        logger.info(df_resultados.to_string(index=False))
        
        return df_resultados
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        nome_modelo: str = "Modelo",
        salvar: bool = True
    ) -> plt.Figure:
        """
        Plota matriz de confusao.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            nome_modelo: Nome do modelo
            salvar: Se True, salva a figura
            
        Returns:
            Figura Matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Nao Alfabetizado', 'Alfabetizado'],
            yticklabels=['Nao Alfabetizado', 'Alfabetizado'],
            ax=ax, annot_kws={'size': 14}
        )
        
        ax.set_xlabel('Predito', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        ax.set_title(f'Matriz de Confusao - {nome_modelo}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if salvar:
            nome_arquivo = f"confusion_matrix_{nome_modelo.lower().replace(' ', '_')}.png"
            fig.savefig(self.output_dir / nome_arquivo, dpi=150, bbox_inches='tight')
            logger.info(f"Salvo: {self.output_dir / nome_arquivo}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        predicoes: Dict[str, Dict],
        salvar: bool = True
    ) -> go.Figure:
        """
        Plota curvas ROC de todos os modelos.
        
        Args:
            y_true: Valores reais
            predicoes: Dicionario {modelo: {y_pred, y_proba}}
            salvar: Se True, salva a figura
            
        Returns:
            Figura Plotly
        """
        logger.info("Gerando curvas ROC...")
        
        cores = ['#2E86AB', '#A23B72', '#28A745', '#FFC107']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random (AUC=0.50)',
            showlegend=True
        ))
        
        for i, (nome, preds) in enumerate(predicoes.items()):
            if 'y_proba' not in preds:
                continue
            
            fpr, tpr, _ = roc_curve(y_true, preds['y_proba'])
            auc = roc_auc_score(y_true, preds['y_proba'])
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color=cores[i % len(cores)], width=2),
                name=f'{nome} (AUC={auc:.3f})',
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='<b>Curvas ROC - Comparativo de Modelos</b>',
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title='Taxa de Falsos Positivos (FPR)',
            yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(gridcolor='#eee', range=[0, 1]),
            yaxis=dict(gridcolor='#eee', range=[0, 1])
        )
        
        if salvar:
            fig.write_html(self.output_dir / "roc_curves.html")
            logger.info(f"Salvo: {self.output_dir / 'roc_curves.html'}")
        
        return fig
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        predicoes: Dict[str, Dict],
        salvar: bool = True
    ) -> go.Figure:
        """
        Plota curvas Precision-Recall.
        
        Args:
            y_true: Valores reais
            predicoes: Dicionario {modelo: {y_pred, y_proba}}
            salvar: Se True, salva a figura
            
        Returns:
            Figura Plotly
        """
        logger.info("Gerando curvas Precision-Recall...")
        
        cores = ['#2E86AB', '#A23B72', '#28A745', '#FFC107']
        
        fig = go.Figure()
        
        for i, (nome, preds) in enumerate(predicoes.items()):
            if 'y_proba' not in preds:
                continue
            
            precision, recall, _ = precision_recall_curve(y_true, preds['y_proba'])
            ap = average_precision_score(y_true, preds['y_proba'])
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                line=dict(color=cores[i % len(cores)], width=2),
                name=f'{nome} (AP={ap:.3f})',
                hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='<b>Curvas Precision-Recall</b>',
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(x=0.1, y=0.1),
            xaxis=dict(gridcolor='#eee', range=[0, 1]),
            yaxis=dict(gridcolor='#eee', range=[0, 1])
        )
        
        if salvar:
            fig.write_html(self.output_dir / "precision_recall_curves.html")
            logger.info(f"Salvo: {self.output_dir / 'precision_recall_curves.html'}")
        
        return fig
    
    def plot_feature_importance(
        self,
        df_importance: pd.DataFrame,
        nome_modelo: str = "Modelo",
        top_n: int = 20,
        salvar: bool = True
    ) -> go.Figure:
        """
        Plota importancia das features.
        
        Args:
            df_importance: DataFrame com importancias
            nome_modelo: Nome do modelo
            top_n: Numero de features a exibir
            salvar: Se True, salva a figura
            
        Returns:
            Figura Plotly
        """
        logger.info(f"Gerando grafico de feature importance ({nome_modelo})...")
        
        top_features = df_importance.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Blues',
                    showscale=False
                )
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f'<b>Top {top_n} Features - {nome_modelo}</b>',
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title='Importancia',
            yaxis_title='Feature',
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(autorange='reversed')
        )
        
        if salvar:
            nome_arquivo = f"feature_importance_{nome_modelo.lower().replace(' ', '_')}.html"
            fig.write_html(self.output_dir / nome_arquivo)
            logger.info(f"Salvo: {self.output_dir / nome_arquivo}")
        
        return fig
    
    def plot_comparativo_metricas(
        self,
        df_resultados: pd.DataFrame,
        salvar: bool = True
    ) -> go.Figure:
        """
        Plota grafico comparativo de metricas entre modelos.
        
        Args:
            df_resultados: DataFrame com resultados
            salvar: Se True, salva a figura
            
        Returns:
            Figura Plotly
        """
        logger.info("Gerando grafico comparativo de metricas...")
        
        metricas = ['accuracy', 'precision', 'recall', 'f1_score']
        metricas_disponiveis = [m for m in metricas if m in df_resultados.columns]
        
        if 'roc_auc' in df_resultados.columns:
            metricas_disponiveis.append('roc_auc')
        
        cores = ['#2E86AB', '#A23B72', '#28A745', '#FFC107']
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(df_resultados.iterrows()):
            fig.add_trace(go.Bar(
                name=row['modelo'],
                x=metricas_disponiveis,
                y=[row[m] for m in metricas_disponiveis],
                marker_color=cores[i % len(cores)],
                text=[f'{row[m]:.3f}' for m in metricas_disponiveis],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=dict(
                text='<b>Comparativo de Metricas por Modelo</b>',
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title='Metrica',
            yaxis_title='Score',
            barmode='group',
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(range=[0, 1.1], gridcolor='#eee'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        if salvar:
            fig.write_html(self.output_dir / "comparativo_metricas.html")
            logger.info(f"Salvo: {self.output_dir / 'comparativo_metricas.html'}")
        
        return fig
    
    def gerar_relatorio_avaliacao(
        self,
        y_true: np.ndarray,
        predicoes: Dict[str, Dict],
        salvar: bool = True
    ) -> Dict[str, Any]:
        """
        Gera relatorio completo de avaliacao.
        
        Args:
            y_true: Valores reais
            predicoes: Dicionario com predicoes
            salvar: Se True, salva figuras
            
        Returns:
            Dicionario com metricas e figuras
        """
        logger.info("Gerando relatorio completo de avaliacao...")
        
        df_resultados = self.avaliar_todos_modelos(y_true, predicoes)
        
        figuras = {}
        
        figuras['roc_curves'] = self.plot_roc_curves(y_true, predicoes, salvar)
        figuras['pr_curves'] = self.plot_precision_recall_curves(y_true, predicoes, salvar)
        figuras['comparativo'] = self.plot_comparativo_metricas(df_resultados, salvar)
        
        for nome, preds in predicoes.items():
            figuras[f'cm_{nome}'] = self.plot_confusion_matrix(
                y_true, preds['y_pred'], nome, salvar
            )
        
        if salvar:
            df_resultados.to_csv(self.output_dir / "metricas_modelos.csv", index=False)
            logger.info(f"Salvo: {self.output_dir / 'metricas_modelos.csv'}")
        
        relatorio = {
            'metricas': df_resultados,
            'figuras': figuras
        }
        
        logger.info("Relatorio de avaliacao gerado!")
        
        return relatorio
