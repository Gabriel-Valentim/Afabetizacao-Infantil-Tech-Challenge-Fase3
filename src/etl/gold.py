"""
Camada Gold - Enriquecimento e feature engineering.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.utils.helpers import resumo_dataframe, log_merge, timer
from config import COLUMN_MAPPING, GOLD_DIR

logger = logging.getLogger(__name__)


class GoldLayer:
    """Camada Gold do pipeline ETL - enriquecimento e features derivadas."""
    
    def __init__(self, column_mapping=COLUMN_MAPPING, output_dir: Path = GOLD_DIR):
        self.column_mapping = column_mapping
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _merge_seguro(
        self,
        df_base: pd.DataFrame,
        df_enrich: pd.DataFrame,
        chave: str,
        nome_merge: str
    ) -> pd.DataFrame:
        """Realiza merge com logging e validacao."""
        if chave not in df_base.columns or chave not in df_enrich.columns:
            logger.warning(f"Merge {nome_merge} pulado - coluna {chave} nao encontrada")
            return df_base
        
        antes = len(df_base)
        df_result = df_base.merge(df_enrich, on=chave, how='left')
        depois = len(df_result)
        
        log_merge(nome_merge, antes, depois, chave, logger)
        logger.info(f"Apos merge {nome_merge}: {len(df_result.columns)} colunas")
        
        return df_result
    
    @timer
    def enriquecer_dados(self, bases_silver: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Enriquece a base de alunos com dados de outras fontes."""
        logger.info("Iniciando enriquecimento de dados...")
        
        df = bases_silver['alunos'].copy()
        logger.info(f"Base inicial: {len(df):,} alunos x {len(df.columns)} colunas")
        
        if 'id_municipio' not in df.columns:
            logger.error("Coluna id_municipio nao encontrada na base de alunos")
            return df
        
        df = self._merge_seguro(df, bases_silver['indicadores'], 'id_municipio', 'Indicadores INSE')
        df = self._merge_seguro(df, bases_silver['censo'], 'id_municipio', 'Censo Escolar')
        df = self._merge_seguro(df, bases_silver['ideb'], 'id_municipio', 'IDEB')
        df = self._merge_seguro(df, bases_silver['populacao'], 'id_municipio', 'Populacao')
        df = self._merge_seguro(df, bases_silver['pib'], 'id_municipio', 'PIB')
        
        logger.info(f"Base enriquecida: {len(df):,} alunos x {len(df.columns)} colunas")
        return df
    
    @timer
    def criar_features_derivadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas para enriquecer a analise."""
        logger.info("Criando features derivadas...")
        
        df = df.copy()
        
        if 'sigla_uf' in df.columns:
            df['nome_regiao'] = df['sigla_uf'].map(self.column_mapping.uf_para_regiao)
            logger.info(f"Coluna nome_regiao criada! Regioes: {df['nome_regiao'].unique().tolist()}")
        
        if 'tipo_dependencia' in df.columns:
            df['desc_dependencia'] = df['tipo_dependencia'].map(self.column_mapping.dependencia_labels)
            logger.info("Coluna desc_dependencia criada!")
        
        col_pop = [c for c in df.columns if c.startswith('populacao_')]
        col_pib = [c for c in df.columns if c.startswith('pib_') and 'per_capita' not in c]
        
        if col_pop and col_pib:
            pop_col = col_pop[0]
            pib_col = col_pib[0]
            df['pib_per_capita'] = df[pib_col] / df[pop_col].replace(0, np.nan)
            logger.info("Coluna pib_per_capita criada!")
        
        colunas_infra = [c for c in df.columns if c.startswith('censo_') and c != 'censo_id_escola']
        
        if colunas_infra:
            df['score_infraestrutura'] = df[colunas_infra].sum(axis=1)
            df['score_infraestrutura_pct'] = (df['score_infraestrutura'] / len(colunas_infra) * 100).round(1)
            logger.info(f"Score infraestrutura criado (baseado em {len(colunas_infra)} indicadores)")
        
        logger.info("Features derivadas criadas!")
        return df
    
    @timer
    def exportar_base(self, df: pd.DataFrame, nome_arquivo: str = "gold_alunos_enriquecido.csv") -> Path:
        """Exporta a base final para arquivo CSV."""
        arquivo_saida = self.output_dir / nome_arquivo
        
        df.to_csv(arquivo_saida, index=False)
        
        tamanho_mb = arquivo_saida.stat().st_size / (1024 * 1024)
        
        logger.info(f"""
BASE EXPORTADA COM SUCESSO!
   Arquivo: {arquivo_saida}
   Tamanho: {tamanho_mb:.2f} MB
   Linhas: {len(df):,}
   Colunas: {len(df.columns)}
""")
        
        return arquivo_saida
    
    @timer
    def processar(
        self,
        bases_silver: Dict[str, pd.DataFrame],
        exportar: bool = True,
        nome_arquivo: str = "gold_alunos_enriquecido.csv"
    ) -> pd.DataFrame:
        """Executa o processamento completo da camada Gold."""
        logger.info("Iniciando processamento da camada Gold...")
        
        df = self.enriquecer_dados(bases_silver)
        df = self.criar_features_derivadas(df)
        
        if exportar:
            self.exportar_base(df, nome_arquivo)
        
        resumo_dataframe(df, "gold_alunos_enriquecido", logger)
        
        return df
