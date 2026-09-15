"""
Geracao de Relatorios de EDA.

Este modulo gera relatorios estatisticos e resumos
dos dados para documentacao e apresentacao.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import FIGURES_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


class EDAReporter:
    """
    Classe para geracao de relatorios de EDA.
    
    Gera relatorios estatisticos em formato texto e DataFrame
    para documentacao e analise.
    
    Attributes:
        output_dir: Diretorio de saida
        target_column: Coluna alvo
    """
    
    def __init__(
        self,
        output_dir: Path = FIGURES_DIR,
        target_column: str = TARGET_COLUMN
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_column = target_column
    
    def resumo_variavel_alvo(self, df: pd.DataFrame) -> Dict:
        """
        Gera resumo estatistico da variavel alvo.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Dicionario com estatisticas
        """
        total = len(df)
        alfabetizados = df[self.target_column].sum()
        nao_alfabetizados = total - alfabetizados
        taxa = alfabetizados / total * 100
        razao = alfabetizados / nao_alfabetizados if nao_alfabetizados > 0 else np.inf
        
        resumo = {
            'total_alunos': total,
            'alfabetizados': alfabetizados,
            'nao_alfabetizados': nao_alfabetizados,
            'taxa_alfabetizacao': round(taxa, 2),
            'razao_classes': round(razao, 2),
            'balanceamento': 'Equilibrado' if 40 < taxa < 60 else 'Desbalanceado'
        }
        
        logger.info(f"""
{'='*60}
RESUMO DA VARIAVEL ALVO
{'='*60}
Total de alunos: {total:,}
Alfabetizados: {alfabetizados:,} ({taxa:.2f}%)
Nao Alfabetizados: {nao_alfabetizados:,} ({100-taxa:.2f}%)
Razao: {razao:.2f}:1
Balanceamento: {resumo['balanceamento']}
""")
        
        return resumo
    
    def resumo_por_uf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera resumo de alfabetizacao por UF.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com resumo por UF
        """
        if 'sigla_uf' not in df.columns:
            logger.warning("Coluna 'sigla_uf' nao encontrada")
            return pd.DataFrame()
        
        resumo = df.groupby('sigla_uf').agg(
            total_alunos=(self.target_column, 'count'),
            alfabetizados=(self.target_column, 'sum')
        ).reset_index()
        
        resumo['nao_alfabetizados'] = resumo['total_alunos'] - resumo['alfabetizados']
        resumo['taxa_alfabetizacao'] = (resumo['alfabetizados'] / resumo['total_alunos'] * 100).round(2)
        resumo['ranking'] = resumo['taxa_alfabetizacao'].rank(ascending=False).astype(int)
        resumo = resumo.sort_values('taxa_alfabetizacao', ascending=False)
        
        media = resumo['taxa_alfabetizacao'].mean()
        resumo['status'] = resumo['taxa_alfabetizacao'].apply(
            lambda x: 'Acima da media' if x >= media else 'Abaixo da media'
        )
        
        return resumo
    
    def resumo_por_regiao(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera resumo de alfabetizacao por regiao.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com resumo por regiao
        """
        if 'nome_regiao' not in df.columns:
            logger.warning("Coluna 'nome_regiao' nao encontrada")
            return pd.DataFrame()
        
        resumo = df.groupby('nome_regiao').agg(
            total_alunos=(self.target_column, 'count'),
            alfabetizados=(self.target_column, 'sum')
        ).reset_index()
        
        resumo['taxa_alfabetizacao'] = (resumo['alfabetizados'] / resumo['total_alunos'] * 100).round(2)
        resumo = resumo.sort_values('taxa_alfabetizacao', ascending=False)
        
        return resumo
    
    def resumo_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera resumo de valores faltantes.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com resumo de missing
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        resumo = pd.DataFrame({
            'coluna': df.columns,
            'missing_count': missing.values,
            'missing_pct': missing_pct.values
        })
        
        resumo = resumo[resumo['missing_count'] > 0].sort_values('missing_pct', ascending=False)
        
        if len(resumo) > 0:
            logger.info(f"""
{'='*60}
ANALISE DE MISSING VALUES
{'='*60}
Colunas com valores faltantes: {len(resumo)}
""")
            logger.info(resumo.head(20).to_string(index=False))
        
        return resumo
    
    def resumo_estatistico(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera resumo estatistico das variaveis numericas.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com estatisticas descritivas
        """
        df_num = df.select_dtypes(include=[np.number])
        
        resumo = df_num.describe().T
        resumo['missing'] = df_num.isnull().sum()
        resumo['missing_pct'] = (resumo['missing'] / len(df) * 100).round(2)
        
        return resumo
    
    def gerar_relatorio_completo(
        self,
        df: pd.DataFrame,
        salvar: bool = True
    ) -> Dict[str, any]:
        """
        Gera relatorio completo de EDA.
        
        Args:
            df: DataFrame com os dados
            salvar: Se True, salva o relatorio
            
        Returns:
            Dicionario com todos os resumos
        """
        logger.info("Gerando relatorio completo de EDA...")
        
        relatorio = {
            'variavel_alvo': self.resumo_variavel_alvo(df),
            'por_uf': self.resumo_por_uf(df),
            'por_regiao': self.resumo_por_regiao(df),
            'missing_values': self.resumo_missing_values(df),
            'estatistico': self.resumo_estatistico(df)
        }
        
        if salvar:
            for nome, dados in relatorio.items():
                if isinstance(dados, pd.DataFrame) and len(dados) > 0:
                    arquivo = self.output_dir / f"resumo_{nome}.csv"
                    dados.to_csv(arquivo, index=False)
                    logger.info(f"Salvo: {arquivo}")
        
        logger.info("Relatorio completo gerado!")
        
        return relatorio
