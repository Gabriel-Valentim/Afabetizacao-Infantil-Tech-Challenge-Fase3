"""
Pipeline ETL completo - Orquestracao das camadas Bronze, Silver e Gold.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .bronze import BronzeLayer
from .silver import SilverLayer
from .gold import GoldLayer
from src.utils.helpers import timer, setup_logging
from config import DADOS_ANTIGOS, DADOS_NOVOS, GOLD_DIR

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Pipeline ETL completo para processamento de dados de alfabetizacao."""
    
    def __init__(
        self,
        dados_antigos: Path = DADOS_ANTIGOS,
        dados_novos: Path = DADOS_NOVOS,
        output_dir: Path = GOLD_DIR,
        ano_referencia: int = 2023
    ):
        self.bronze = BronzeLayer(dados_antigos, dados_novos)
        self.silver = SilverLayer(ano_referencia=ano_referencia)
        self.gold = GoldLayer(output_dir=output_dir)
        
        self._bases_bronze: Optional[Dict[str, pd.DataFrame]] = None
        self._bases_silver: Optional[Dict[str, pd.DataFrame]] = None
        self._df_gold: Optional[pd.DataFrame] = None
    
    @property
    def bases_bronze(self) -> Optional[Dict[str, pd.DataFrame]]:
        return self._bases_bronze
    
    @property
    def bases_silver(self) -> Optional[Dict[str, pd.DataFrame]]:
        return self._bases_silver
    
    @property
    def df_gold(self) -> Optional[pd.DataFrame]:
        return self._df_gold
    
    @timer
    def executar_bronze(self) -> Dict[str, pd.DataFrame]:
        """Executa a camada Bronze - Ingestao de dados."""
        logger.info("=" * 60)
        logger.info("CAMADA BRONZE - Ingestao de Dados")
        logger.info("=" * 60)
        
        self._bases_bronze = self.bronze.carregar_todas_bases()
        return self._bases_bronze
    
    @timer
    def executar_silver(self, bases_bronze: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """Executa a camada Silver - Limpeza e padronizacao."""
        logger.info("=" * 60)
        logger.info("CAMADA SILVER - Limpeza e Padronizacao")
        logger.info("=" * 60)
        
        if bases_bronze is None:
            if self._bases_bronze is None:
                raise ValueError("Camada Bronze nao executada. Execute executar_bronze() primeiro.")
            bases_bronze = self._bases_bronze
        
        self._bases_silver = self.silver.processar_todas_bases(bases_bronze)
        return self._bases_silver
    
    @timer
    def executar_gold(
        self,
        bases_silver: Optional[Dict[str, pd.DataFrame]] = None,
        exportar: bool = True,
        nome_arquivo: str = "gold_alunos_enriquecido.csv"
    ) -> pd.DataFrame:
        """Executa a camada Gold - Enriquecimento e features."""
        logger.info("=" * 60)
        logger.info("CAMADA GOLD - Enriquecimento e Features")
        logger.info("=" * 60)
        
        if bases_silver is None:
            if self._bases_silver is None:
                raise ValueError("Camada Silver nao executada. Execute executar_silver() primeiro.")
            bases_silver = self._bases_silver
        
        self._df_gold = self.gold.processar(bases_silver, exportar=exportar, nome_arquivo=nome_arquivo)
        return self._df_gold
    
    @timer
    def executar_completo(
        self,
        exportar: bool = True,
        nome_arquivo: str = "gold_alunos_enriquecido.csv"
    ) -> pd.DataFrame:
        """Executa o pipeline ETL completo (Bronze -> Silver -> Gold)."""
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE ETL COMPLETO")
        logger.info("=" * 60)
        
        self.executar_bronze()
        self.executar_silver()
        df_gold = self.executar_gold(exportar=exportar, nome_arquivo=nome_arquivo)
        
        logger.info("=" * 60)
        logger.info("PIPELINE ETL FINALIZADO COM SUCESSO!")
        logger.info("=" * 60)
        
        return df_gold
    
    def carregar_gold_existente(self, arquivo: Optional[Path] = None) -> pd.DataFrame:
        """Carrega uma base Gold existente do disco."""
        if arquivo is None:
            arquivo = self.gold.output_dir / "gold_alunos_enriquecido.csv"
        
        logger.info(f"Carregando base Gold existente: {arquivo}")
        
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo}")
        
        self._df_gold = pd.read_csv(arquivo)
        logger.info(f"Base carregada: {len(self._df_gold):,} linhas x {len(self._df_gold.columns)} colunas")
        
        return self._df_gold


def main():
    """Funcao principal para execucao do pipeline ETL."""
    setup_logging()
    pipeline = ETLPipeline()
    df_gold = pipeline.executar_completo(exportar=True)
    return df_gold


if __name__ == "__main__":
    main()
