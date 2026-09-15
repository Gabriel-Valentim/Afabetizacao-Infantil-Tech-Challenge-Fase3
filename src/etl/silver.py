"""
Camada Silver - Limpeza e padronizacao de dados.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.helpers import (
    padronizar_nomes_colunas,
    converter_id_para_int64,
    resumo_dataframe,
    remover_colunas_anos_antigos,
    timer
)
from config import COLUMN_MAPPING, FEATURE_CONFIG

logger = logging.getLogger(__name__)


class SilverLayer:
    """Camada Silver do pipeline ETL - limpeza e padronizacao."""
    
    def __init__(
        self,
        ano_referencia: int = 2023,
        column_mapping=COLUMN_MAPPING,
        feature_config=FEATURE_CONFIG
    ):
        self.ano_referencia = ano_referencia
        self.column_mapping = column_mapping
        self.feature_config = feature_config
    
    @timer
    def processar_alunos(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa a base de alunos da camada Bronze."""
        logger.info("Processando base de alunos...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df, self.column_mapping.colunas_padronizadas)
        df = converter_id_para_int64(df, ['id_municipio', 'id_escola', 'cod_uf'])
        
        if 'tipo_dependencia' in df.columns:
            antes = len(df)
            df = df[df['tipo_dependencia'].isin([1, 2, 3, 4])]
            logger.info(f"Filtro dependencia: {antes:,} -> {len(df):,} registros")
        
        if 'ano_avaliacao' in df.columns and 'id_aluno' in df.columns:
            antes = len(df)
            df = df.drop_duplicates(subset=['ano_avaliacao', 'id_aluno'], keep='first')
            logger.info(f"Remocao duplicatas: {antes:,} -> {len(df):,} registros")
        
        df = remover_colunas_anos_antigos(df, str(self.ano_referencia), logger)
        
        resumo_dataframe(df, "silver_alunos", logger)
        return df
    
    @timer
    def processar_indicadores_inse(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa indicadores educacionais INSE."""
        logger.info("Processando indicadores INSE...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df)
        df = converter_id_para_int64(df, ['id_municipio', 'id_escola', 'ano'])
        
        if 'ano' in df.columns:
            df = df[df['ano'] == self.ano_referencia]
        
        colunas_numericas = [
            'inse', 'quantidade_alunos_inse',
            'percentual_nivel_1', 'percentual_nivel_2', 'percentual_nivel_3',
            'percentual_nivel_4', 'percentual_nivel_5', 'percentual_nivel_6',
            'percentual_nivel_7', 'percentual_nivel_8'
        ]
        colunas_disponiveis = [c for c in colunas_numericas if c in df.columns]
        
        if 'id_municipio' in df.columns and colunas_disponiveis:
            df = df.groupby('id_municipio')[colunas_disponiveis].mean().reset_index()
            
            colunas_renomear = {
                c: f'inse_{c}' if not c.startswith('inse') and c != 'id_municipio' else c 
                for c in df.columns
            }
            df = df.rename(columns=colunas_renomear)
            
            logger.info(f"Indicadores INSE: {len(df):,} municipios (media por municipio)")
        else:
            df = pd.DataFrame({'id_municipio': pd.array([], dtype='Int64')})
            logger.warning("Indicadores INSE: Dados insuficientes para agregacao")
        
        return df
    
    @timer
    def processar_censo_escolar(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa dados do Censo Escolar."""
        logger.info("Processando censo escolar...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df)
        df = converter_id_para_int64(df, ['id_municipio', 'id_escola', 'ano'])
        
        if 'ano' in df.columns:
            df = df[df['ano'] == self.ano_referencia]
        
        colunas_censo = [
            'id_municipio', 'agua_filtrada', 'agua_potavel', 'energia_rede_publica',
            'biblioteca', 'laboratorio_informatica', 'quadra_esportes',
            'sala_leitura', 'parque_infantil', 'patio_coberto',
            'internet', 'banda_larga', 'alimentacao'
        ]
        colunas_disponiveis = [c for c in colunas_censo if c in df.columns]
        colunas_num = [c for c in colunas_disponiveis if c != 'id_municipio']
        
        if 'id_municipio' in df.columns and colunas_num:
            df = df.groupby('id_municipio')[colunas_num].mean().reset_index()
            colunas_renomear = {c: f'censo_{c}' for c in df.columns if c != 'id_municipio'}
            df = df.rename(columns=colunas_renomear)
            logger.info(f"Censo Escolar: {len(df):,} municipios (media por municipio)")
        else:
            df = pd.DataFrame({'id_municipio': pd.array([], dtype='Int64')})
            logger.warning("Censo Escolar: Dados insuficientes para agregacao")
        
        return df
    
    @timer
    def processar_ideb(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa dados do IDEB."""
        logger.info("Processando IDEB...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df)
        df = converter_id_para_int64(df, ['id_municipio', 'ano'])
        
        if 'anos_escolares' in df.columns:
            mask_iniciais = df['anos_escolares'].str.contains('iniciais', case=False, na=False)
            df = df[mask_iniciais].copy()
        
        colunas_ideb = [
            'id_municipio', 'ideb', 'nota_saeb_matematica',
            'nota_saeb_lingua_portuguesa', 'taxa_aprovacao'
        ]
        colunas_disponiveis = [c for c in colunas_ideb if c in df.columns]
        df = df[colunas_disponiveis].dropna(subset=['id_municipio'])
        
        colunas_num = [c for c in colunas_disponiveis if c != 'id_municipio']
        
        if len(df) > 0 and colunas_num:
            df = df.groupby('id_municipio')[colunas_num].mean().reset_index()
            colunas_renomear = {c: f'ideb_{c}' for c in df.columns if c != 'id_municipio'}
            df = df.rename(columns=colunas_renomear)
            logger.info(f"IDEB: {len(df):,} municipios")
        else:
            df = pd.DataFrame({'id_municipio': pd.array([], dtype='Int64')})
            logger.warning("IDEB: Dados insuficientes para merge")
        
        return df
    
    @timer
    def processar_populacao(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa dados de populacao."""
        logger.info("Processando populacao...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df)
        df = converter_id_para_int64(df, ['id_municipio', 'ano'])
        
        if 'ano' in df.columns:
            df = df[df['ano'] == self.ano_referencia]
        
        df = df[['id_municipio', 'populacao']]
        df = df.rename(columns={'populacao': f'populacao_{self.ano_referencia}'})
        
        logger.info(f"Populacao: {len(df):,} municipios")
        return df
    
    @timer
    def processar_pib(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Processa dados de PIB."""
        logger.info("Processando PIB...")
        
        df = df_bronze.copy()
        df = padronizar_nomes_colunas(df)
        df = converter_id_para_int64(df, ['id_municipio', 'ano'])
        
        if 'ano' in df.columns:
            df = df[df['ano'] == self.ano_referencia]
        
        df = df[['id_municipio', 'pib']]
        df = df.rename(columns={'pib': f'pib_{self.ano_referencia}'})
        
        logger.info(f"PIB: {len(df):,} municipios")
        return df
    
    @timer
    def processar_todas_bases(self, bases_bronze: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Processa todas as bases da camada Bronze."""
        logger.info("Processando todas as bases para Silver...")
        
        bases_silver = {
            'alunos': self.processar_alunos(bases_bronze['alunos']),
            'indicadores': self.processar_indicadores_inse(bases_bronze['indicadores']),
            'censo': self.processar_censo_escolar(bases_bronze['censo']),
            'ideb': self.processar_ideb(bases_bronze['ideb']),
            'populacao': self.processar_populacao(bases_bronze['populacao']),
            'pib': self.processar_pib(bases_bronze['pib'])
        }
        
        logger.info("Todas as bases processadas para Silver!")
        return bases_silver
