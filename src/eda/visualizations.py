"""
Visualizacoes para Analise Exploratoria de Dados.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import VIZ_CONFIG, FIGURES_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


class EDAVisualizer:
    """Classe para visualizacoes de EDA."""
    
    def __init__(
        self,
        viz_config=VIZ_CONFIG,
        output_dir: Path = FIGURES_DIR,
        target_column: str = TARGET_COLUMN
    ):
        self.viz_config = viz_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_column = target_column
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
    
    def plot_distribuicao_target(self, df: pd.DataFrame, salvar: bool = True) -> go.Figure:
        """Plota distribuicao da variavel alvo com grafico de barras e pizza."""
        logger.info("Gerando grafico de distribuicao da variavel alvo...")
        
        cores = self.viz_config.cores
        
        total = len(df)
        alfabetizados = df[self.target_column].sum()
        nao_alfabetizados = total - alfabetizados
        taxa = alfabetizados / total * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=('Distribuicao Absoluta', 'Distribuicao Percentual'),
            horizontal_spacing=0.15
        )
        
        fig.add_trace(
            go.Bar(
                x=['Nao Alfabetizado', 'Alfabetizado'],
                y=[nao_alfabetizados, alfabetizados],
                marker=dict(
                    color=[cores['nao_alfabetizado'], cores['alfabetizado']],
                    line=dict(color='white', width=2)
                ),
                text=[
                    f'{nao_alfabetizados:,}<br>({100-taxa:.1f}%)',
                    f'{alfabetizados:,}<br>({taxa:.1f}%)'
                ],
                textposition='outside',
                textfont=dict(size=13, color='#333'),
                showlegend=False,
                width=0.6
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(
                labels=['Alfabetizado', 'Nao Alfabetizado'],
                values=[alfabetizados, nao_alfabetizados],
                marker=dict(
                    colors=[cores['alfabetizado'], cores['nao_alfabetizado']],
                    line=dict(color='white', width=3)
                ),
                textinfo='percent+label',
                textfont=dict(size=13, color='white'),
                textposition='inside',
                hole=0.45,
                pull=[0.03, 0.03],
                rotation=90
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(text='Distribuicao da Variavel Alvo: Alfabetizacao', font=dict(size=22, color='#1a1a2e'), x=0.5, y=0.95),
            height=520,
            margin=dict(t=100, b=80, l=60, r=60),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        fig.update_yaxes(range=[0, max(nao_alfabetizados, alfabetizados) * 1.25], row=1, col=1, gridcolor='#eee', tickformat=',')
        
        if salvar:
            fig.write_html(self.output_dir / "distribuicao_target.html")
            logger.info(f"Salvo em: {self.output_dir / 'distribuicao_target.html'}")
        
        return fig
    
    def plot_taxa_por_uf(self, df: pd.DataFrame, salvar: bool = True) -> go.Figure:
        """Plota taxa de alfabetizacao por Unidade Federativa."""
        logger.info("Gerando grafico de taxa por UF...")
        
        cores = self.viz_config.cores
        
        taxa_uf = df.groupby('sigla_uf').agg(
            total_alunos=(self.target_column, 'count'),
            alfabetizados=(self.target_column, 'sum')
        ).reset_index()
        taxa_uf['taxa'] = (taxa_uf['alfabetizados'] / taxa_uf['total_alunos'] * 100).round(2)
        taxa_uf = taxa_uf.sort_values('taxa', ascending=False)
        
        media = taxa_uf['taxa'].mean()
        
        cores_uf = [cores['alfabetizado'] if x >= media else cores['nao_alfabetizado'] for x in taxa_uf['taxa']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=taxa_uf['sigla_uf'],
            y=taxa_uf['taxa'],
            marker=dict(color=cores_uf, line=dict(color='white', width=2)),
            text=taxa_uf['taxa'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=12, color='#333'),
            hovertemplate='%{x}<br>Taxa: %{y:.1f}%<br>Alunos: %{customdata:,}<extra></extra>',
            customdata=taxa_uf['total_alunos']
        ))
        
        fig.add_hline(y=media, line_dash="dash", line_color=cores['destaque'], line_width=3,
                     annotation_text=f"Media: {media:.1f}%", annotation_position="top right",
                     annotation_font_size=12, annotation_font_color=cores['destaque'])
        
        fig.update_layout(
            title=dict(text='Taxa de Alfabetizacao por Unidade Federativa', font=dict(size=20, color='#1a1a2e'), x=0.5, y=0.95),
            xaxis_title='Estado',
            yaxis_title='Taxa de Alfabetizacao (%)',
            height=500,
            margin=dict(t=100, b=80, l=60, r=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(gridcolor='#E5E5E5', range=[0, max(taxa_uf['taxa']) * 1.15], ticksuffix='%', dtick=10),
            bargap=0.3
        )
        
        if salvar:
            fig.write_html(self.output_dir / "taxa_por_uf.html")
            logger.info(f"Salvo em: {self.output_dir / 'taxa_por_uf.html'}")
        
        return fig
    
    def plot_taxa_por_regiao(self, df: pd.DataFrame, salvar: bool = True) -> go.Figure:
        """Plota taxa de alfabetizacao por regiao."""
        logger.info("Gerando grafico de taxa por regiao...")
        
        if 'nome_regiao' not in df.columns:
            logger.warning("Coluna 'nome_regiao' nao encontrada")
            return None
        
        taxa_regiao = df.groupby('nome_regiao').agg(
            total_alunos=(self.target_column, 'count'),
            alfabetizados=(self.target_column, 'sum')
        ).reset_index()
        taxa_regiao['taxa'] = (taxa_regiao['alfabetizados'] / taxa_regiao['total_alunos'] * 100).round(2)
        taxa_regiao = taxa_regiao.sort_values('taxa', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=taxa_regiao['nome_regiao'],
                y=taxa_regiao['taxa'],
                marker=dict(color=self.viz_config.paleta_regioes[:len(taxa_regiao)]),
                text=taxa_regiao['taxa'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                textfont=dict(size=14, color='black'),
                hovertemplate='%{x}<br>Taxa: %{y:.1f}%<br>Alunos: %{customdata:,}<extra></extra>',
                customdata=taxa_regiao['total_alunos']
            )
        ])
        
        fig.update_layout(
            title=dict(text='Taxa de Alfabetizacao por Regiao', font=dict(size=20, color='#1a1a2e'), x=0.5, y=0.95),
            xaxis_title='Regiao',
            yaxis_title='Taxa de Alfabetizacao (%)',
            height=450,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(gridcolor='#E5E5E5', range=[0, 100], ticksuffix='%')
        )
        
        if salvar:
            fig.write_html(self.output_dir / "taxa_por_regiao.html")
            logger.info(f"Salvo em: {self.output_dir / 'taxa_por_regiao.html'}")
        
        return fig
    
    def plot_taxa_por_dependencia(self, df: pd.DataFrame, salvar: bool = True) -> go.Figure:
        """Plota taxa de alfabetizacao por dependencia administrativa."""
        logger.info("Gerando grafico de taxa por dependencia administrativa...")
        
        col_dep = 'desc_dependencia' if 'desc_dependencia' in df.columns else 'tipo_dependencia'
        
        taxa_dep = df.groupby(col_dep).agg(
            total_alunos=(self.target_column, 'count'),
            alfabetizados=(self.target_column, 'sum')
        ).reset_index()
        taxa_dep['taxa'] = (taxa_dep['alfabetizados'] / taxa_dep['total_alunos'] * 100).round(2)
        taxa_dep = taxa_dep.sort_values('taxa', ascending=False)
        
        cores_dep = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        fig = go.Figure(data=[
            go.Bar(
                x=taxa_dep[col_dep],
                y=taxa_dep['taxa'],
                marker=dict(color=cores_dep[:len(taxa_dep)]),
                text=taxa_dep['taxa'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                textfont=dict(size=14),
                hovertemplate='%{x}<br>Taxa: %{y:.1f}%<br>Alunos: %{customdata:,}<extra></extra>',
                customdata=taxa_dep['total_alunos']
            )
        ])
        
        fig.update_layout(
            title=dict(text='Taxa de Alfabetizacao por Dependencia Administrativa', font=dict(size=20, color='#1a1a2e'), x=0.5, y=0.95),
            xaxis_title='Dependencia Administrativa',
            yaxis_title='Taxa de Alfabetizacao (%)',
            height=450,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(gridcolor='#E5E5E5', range=[0, 100], ticksuffix='%')
        )
        
        if salvar:
            fig.write_html(self.output_dir / "taxa_por_dependencia.html")
            logger.info(f"Salvo em: {self.output_dir / 'taxa_por_dependencia.html'}")
        
        return fig
    
    def plot_heatmap_correlacao(self, df: pd.DataFrame, colunas: Optional[List[str]] = None, salvar: bool = True) -> plt.Figure:
        """Plota heatmap de correlacao entre variaveis numericas."""
        logger.info("Gerando heatmap de correlacao...")
        
        if colunas is None:
            df_num = df.select_dtypes(include=[np.number])
        else:
            df_num = df[colunas]
        
        if len(df_num.columns) > 20:
            if self.target_column in df_num.columns:
                corr_target = df_num.corrwith(df_num[self.target_column]).abs()
                top_cols = corr_target.nlargest(19).index.tolist()
                if self.target_column not in top_cols:
                    top_cols.append(self.target_column)
                df_num = df_num[top_cols]
        
        corr_matrix = df_num.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax,
            annot_kws={'size': 8}
        )
        
        ax.set_title('Matriz de Correlacao', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if salvar:
            fig.savefig(self.output_dir / "heatmap_correlacao.png", dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Salvo em: {self.output_dir / 'heatmap_correlacao.png'}")
        
        return fig
    
    def plot_distribuicao_proficiencia(self, df: pd.DataFrame, salvar: bool = True) -> go.Figure:
        """Plota distribuicao da proficiencia por status de alfabetizacao."""
        logger.info("Gerando grafico de distribuicao da proficiencia...")
        
        if 'proficiencia_lp' not in df.columns:
            logger.warning("Coluna 'proficiencia_lp' nao encontrada")
            return None
        
        cores = self.viz_config.cores
        
        df_valid = df[['proficiencia_lp', self.target_column]].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df_valid[df_valid[self.target_column] == 0]['proficiencia_lp'],
            name='Nao Alfabetizado',
            marker_color=cores['nao_alfabetizado'],
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=df_valid[df_valid[self.target_column] == 1]['proficiencia_lp'],
            name='Alfabetizado',
            marker_color=cores['alfabetizado'],
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title=dict(text='Distribuicao da Proficiencia em Lingua Portuguesa', font=dict(size=20, color='#1a1a2e'), x=0.5, y=0.95),
            xaxis_title='Proficiencia LP',
            yaxis_title='Frequencia',
            barmode='overlay',
            height=450,
            paper_bgcolor='white',
            plot_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        if salvar:
            fig.write_html(self.output_dir / "distribuicao_proficiencia.html")
            logger.info(f"Salvo em: {self.output_dir / 'distribuicao_proficiencia.html'}")
        
        return fig
    
    def gerar_todos_graficos(self, df: pd.DataFrame, salvar: bool = True) -> Dict[str, any]:
        """Gera todos os graficos de EDA."""
        logger.info("Gerando todos os graficos de EDA...")
        
        figuras = {}
        
        figuras['distribuicao_target'] = self.plot_distribuicao_target(df, salvar)
        
        if 'sigla_uf' in df.columns:
            figuras['taxa_por_uf'] = self.plot_taxa_por_uf(df, salvar)
        
        if 'nome_regiao' in df.columns:
            figuras['taxa_por_regiao'] = self.plot_taxa_por_regiao(df, salvar)
        
        if 'tipo_dependencia' in df.columns or 'desc_dependencia' in df.columns:
            figuras['taxa_por_dependencia'] = self.plot_taxa_por_dependencia(df, salvar)
        
        figuras['heatmap_correlacao'] = self.plot_heatmap_correlacao(df, salvar=salvar)
        
        if 'proficiencia_lp' in df.columns:
            figuras['distribuicao_proficiencia'] = self.plot_distribuicao_proficiencia(df, salvar)
        
        logger.info(f"{len(figuras)} graficos gerados!")
        
        return figuras
