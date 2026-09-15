"""
Camada Bronze - Ingestao de dados brutos.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.utils.helpers import resumo_dataframe, timer

logger = logging.getLogger(__name__)


class BronzeLayer:
    """Camada Bronze do pipeline ETL - carrega dados das fontes originais."""
    
    def __init__(self, dados_antigos: Path, dados_novos: Path):
        self.dados_antigos = Path(dados_antigos)
        self.dados_novos = Path(dados_novos)
        self._validate_paths()
        
    def _validate_paths(self) -> None:
        if not self.dados_antigos.exists():
            logger.warning(f"Diretorio nao encontrado: {self.dados_antigos}")
        if not self.dados_novos.exists():
            logger.warning(f"Diretorio nao encontrado: {self.dados_novos}")
    
    @timer
    def carregar_base_alunos(self) -> pd.DataFrame:
        """Carrega a base principal de alunos (TS_ALUNO.csv)."""
        arquivo = self.dados_antigos / "TS_ALUNO.csv"
        
        logger.info(f"Carregando base de alunos: {arquivo}")
        
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo}")
        
        df = pd.read_csv(
            arquivo,
            sep=';',
            encoding='latin-1',
            engine='python',
            on_bad_lines='warn'
        )
        
        if 'CO_MUNICIPIO' in df.columns and 'id_municipio' not in df.columns:
            df = df.rename(columns={'CO_MUNICIPIO': 'id_municipio'})
        
        resumo_dataframe(df, "bronze_alunos", logger)
        
        if 'SG_UF' in df.columns:
            logger.info(f"UFs disponiveis: {df['SG_UF'].nunique()} estados")
        
        return df
    
    @timer
    def carregar_populacao(self) -> pd.DataFrame:
        """Carrega dados de populacao por municipio (IBGE)."""
        arquivo = self.dados_novos / "ibge_populacao_municipio.csv"
        logger.info(f"Carregando populacao: {arquivo}")
        df = pd.read_csv(arquivo)
        logger.info(f"Populacao: {len(df):,} registros")
        return df
    
    @timer
    def carregar_pib(self) -> pd.DataFrame:
        """Carrega dados de PIB por municipio (IBGE)."""
        arquivo = self.dados_novos / "ibge_pib_municipio.csv"
        logger.info(f"Carregando PIB: {arquivo}")
        df = pd.read_csv(arquivo)
        logger.info(f"PIB: {len(df):,} registros")
        return df
    
    @timer
    def carregar_indicadores_educacionais(self) -> pd.DataFrame:
        """Carrega indicadores educacionais INSE."""
        arquivo = self.dados_novos / "inep_indicadores_educacionais_escola_inse.csv"
        logger.info(f"Carregando indicadores educacionais: {arquivo}")
        df = pd.read_csv(arquivo)
        logger.info(f"Indicadores Educacionais: {len(df):,} registros")
        return df
    
    @timer
    def carregar_censo_escolar(self) -> pd.DataFrame:
        """Carrega dados do Censo Escolar."""
        arquivo = self.dados_novos / "censo_escolar_escola.csv"
        logger.info(f"Carregando censo escolar: {arquivo}")
        df = pd.read_csv(arquivo, low_memory=False)
        logger.info(f"Censo Escolar: {len(df):,} registros")
        return df
    
    @timer
    def carregar_ideb(self) -> pd.DataFrame:
        """Carrega dados do IDEB por municipio."""
        arquivo = self.dados_novos / "inep_ideb_municipio.csv"
        logger.info(f"Carregando IDEB: {arquivo}")
        df = pd.read_csv(arquivo)
        logger.info(f"IDEB: {len(df):,} registros")
        return df
    
    @timer
    def carregar_todas_bases(self) -> Dict[str, pd.DataFrame]:
        """Carrega todas as bases de dados da camada Bronze."""
        logger.info("Carregando todas as bases de dados...")
        
        bases = {
            'alunos': self.carregar_base_alunos(),
            'populacao': self.carregar_populacao(),
            'pib': self.carregar_pib(),
            'indicadores': self.carregar_indicadores_educacionais(),
            'censo': self.carregar_censo_escolar(),
            'ideb': self.carregar_ideb()
        }
        
        logger.info("Todas as bases carregadas!")
        return bases
