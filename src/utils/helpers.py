"""
Funcoes auxiliares para o projeto de predicao de alfabetizacao.
"""

import logging
import re
import time
from functools import wraps
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np


def setup_logging(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """Configura o logging do projeto."""
    logging.basicConfig(level=level, format=format_str, datefmt=datefmt)
    return logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator para medir o tempo de execucao de uma funcao."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        logger.info(f"Iniciando: {func.__name__}")
        
        result = func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f}s"
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.2f}s"
            
        logger.info(f"Finalizado: {func.__name__} em {time_str}")
        return result
    return wrapper


def padronizar_nomes_colunas(
    df: pd.DataFrame, 
    mapeamento: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Padroniza os nomes das colunas de um DataFrame."""
    df = df.copy()
    
    if mapeamento:
        colunas_existentes = {k: v for k, v in mapeamento.items() if k in df.columns}
        df = df.rename(columns=colunas_existentes)
    
    novas_colunas = {col: col.lower().strip() for col in df.columns}
    return df.rename(columns=novas_colunas)


def converter_id_para_int64(
    df: pd.DataFrame, 
    colunas: List[str]
) -> pd.DataFrame:
    """Converte colunas de ID para o tipo Int64 (nullable integer)."""
    df = df.copy()
    
    for col in colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
    return df


def resumo_dataframe(
    df: pd.DataFrame, 
    nome: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Gera um resumo estatistico de um DataFrame."""
    memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    stats = {
        'nome': nome,
        'linhas': len(df),
        'colunas': df.shape[1],
        'memoria_mb': round(memoria_mb, 2),
        'tipos_dados': df.dtypes.value_counts().to_dict(),
        'missing_total': df.isnull().sum().sum(),
        'missing_pct': round(df.isnull().sum().sum() / (len(df) * df.shape[1]) * 100, 2)
    }
    
    output = f"""
{'='*60}
RESUMO: {nome}
{'='*60}
Linhas: {stats['linhas']:,}
Colunas: {stats['colunas']}
Memoria: {stats['memoria_mb']:.2f} MB
Missing Total: {stats['missing_total']:,} ({stats['missing_pct']:.2f}%)
"""
    
    if logger:
        logger.info(output)
    else:
        print(output)
        
    return stats


def log_merge(
    nome_merge: str, 
    antes: int, 
    depois: int, 
    chave: str,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """Loga informacoes sobre uma operacao de merge."""
    diff = depois - antes
    pct_change = ((depois - antes) / antes * 100) if antes > 0 else 0
    
    stats = {
        'nome': nome_merge,
        'chave': chave,
        'linhas_antes': antes,
        'linhas_depois': depois,
        'diferenca': diff,
        'pct_mudanca': round(pct_change, 2)
    }
    
    output = f"""
MERGE: {nome_merge}
   Chave: {chave}
   Linhas antes: {antes:,}
   Linhas depois: {depois:,}
   Diferenca: {diff:+,} ({pct_change:+.2f}%)
"""
    
    if logger:
        logger.info(output)
    else:
        print(output)
        
    return stats


def remover_colunas_anos_antigos(
    df: pd.DataFrame, 
    ano_manter: str = '2023',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Remove colunas que contem anos diferentes do especificado."""
    df = df.copy()
    colunas_remover = []
    padrao_ano = re.compile(r'_(20\d{2})$')
    
    for col in df.columns:
        match = padrao_ano.search(col)
        if match:
            ano = match.group(1)
            if ano != ano_manter:
                colunas_remover.append(col)
    
    if colunas_remover:
        msg = f"\nRemovendo {len(colunas_remover)} colunas com anos diferentes de {ano_manter}"
        df = df.drop(columns=colunas_remover)
        
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    return df


def validar_dataframe(
    df: pd.DataFrame,
    colunas_obrigatorias: Optional[List[str]] = None,
    min_linhas: int = 0,
    nome: str = "DataFrame"
) -> bool:
    """Valida um DataFrame contra criterios minimos."""
    erros = []
    
    if df is None:
        erros.append(f"{nome} e None")
    elif len(df) < min_linhas:
        erros.append(f"{nome} tem {len(df)} linhas (minimo: {min_linhas})")
    
    if colunas_obrigatorias:
        colunas_faltantes = [c for c in colunas_obrigatorias if c not in df.columns]
        if colunas_faltantes:
            erros.append(f"Colunas faltantes em {nome}: {colunas_faltantes}")
    
    if erros:
        raise ValueError("\n".join(erros))
    
    return True


def otimizar_tipos_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """Otimiza os tipos de dados para reduzir uso de memoria."""
    df = df.copy()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
                
        elif col_type in ['int64', 'Int64']:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype('Int8')
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype('Int16')
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype('Int32')
                
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
    
    return df
