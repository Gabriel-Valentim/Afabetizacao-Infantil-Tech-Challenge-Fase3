"""
Pipeline Principal de Analise de Alfabetizacao.

Este script executa o pipeline completo de ML:
1. ETL (Bronze -> Silver -> Gold)
2. Analise Exploratoria (EDA)
3. Feature Engineering
4. Treinamento de Modelos
5. Avaliacao e Comparacao
6. Interpretabilidade (SHAP)

Uso:
    python main.py [--etapa ETAPA] [--skip-eda] [--skip-shap]
    
Argumentos:
    --etapa: Etapa especifica para executar (etl, eda, features, train, eval, shap, all)
    --skip-eda: Pular analise exploratoria
    --skip-shap: Pular analise SHAP
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DADOS_ANTIGOS, DADOS_NOVOS,
    DATA_DIR, MODELS_DIR, FIGURES_DIR,
    TARGET_COLUMN, RANDOM_STATE
)
from src.utils.helpers import setup_logging, timer
from src.etl import ETLPipeline
from src.features import FeatureEngineer, FeatureSelector
from src.eda import EDAVisualizer, EDAReporter
from src.modeling import ModelTrainer, ModelEvaluator
from src.interpretability import SHAPAnalyzer


def configurar_argumentos() -> argparse.Namespace:
    """Configura argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Pipeline de Analise de Alfabetizacao'
    )
    parser.add_argument(
        '--etapa',
        type=str,
        default='all',
        choices=['etl', 'eda', 'features', 'train', 'eval', 'shap', 'all'],
        help='Etapa especifica para executar'
    )
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Pular analise exploratoria'
    )
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Pular analise SHAP'
    )
    
    return parser.parse_args()


@timer
def executar_etl(logger: logging.Logger) -> pd.DataFrame:
    """
    Executa pipeline ETL completo.
    
    Returns:
        DataFrame processado (camada Gold)
    """
    logger.info("=" * 70)
    logger.info("ETAPA 1: ETL PIPELINE")
    logger.info("=" * 70)
    
    etl = ETLPipeline()
    
    # Ingerir dados
    logger.info("Ingerindo dados...")
    etl.ingerir_dados(DADOS_ANTIGOS, DADOS_NOVOS)
    
    # Processar pipeline completo
    logger.info("Processando pipeline Bronze -> Silver -> Gold...")
    df_gold = etl.executar_pipeline_completo()
    
    logger.info(f"ETL concluido! Shape final: {df_gold.shape}")
    
    return df_gold


@timer
def executar_eda(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Executa analise exploratoria de dados.
    
    Args:
        df: DataFrame para analise
    """
    logger.info("=" * 70)
    logger.info("ETAPA 2: ANALISE EXPLORATORIA (EDA)")
    logger.info("=" * 70)
    
    # Visualizacoes
    visualizer = EDAVisualizer()
    
    logger.info("Gerando visualizacoes...")
    visualizer.plot_distribuicao_target(df)
    visualizer.plot_taxa_por_uf(df)
    visualizer.plot_taxa_por_regiao(df)
    visualizer.plot_correlacao_heatmap(df)
    
    # Relatorios
    reporter = EDAReporter()
    
    logger.info("Gerando relatorios...")
    reporter.gerar_relatorio_completo(df)
    
    logger.info("EDA concluido!")


@timer
def executar_feature_engineering(
    df: pd.DataFrame, 
    logger: logging.Logger
) -> tuple:
    """
    Executa feature engineering e selecao.
    
    Args:
        df: DataFrame com dados
        
    Returns:
        Tuple com (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 70)
    logger.info("ETAPA 3: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    engineer = FeatureEngineer()
    
    # Preparar dados
    logger.info("Preparando dados...")
    X, y = engineer.preparar_dados(df)
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Amostras: {len(y)}")
    
    # Split treino/teste
    logger.info("Dividindo dados em treino/teste...")
    X_train, X_test, y_train, y_test = engineer.split_train_test(X, y)
    
    logger.info(f"Treino: {len(y_train)} amostras")
    logger.info(f"Teste: {len(y_test)} amostras")
    
    # Selecao de features (opcional)
    selector = FeatureSelector()
    
    logger.info("Analisando importancia das features...")
    df_importance = selector.calcular_importancia_rf(X_train, y_train)
    df_importance.to_csv(FIGURES_DIR / "feature_importance_rf.csv", index=False)
    
    logger.info("Feature engineering concluido!")
    
    return X_train, X_test, y_train, y_test


@timer
def executar_treinamento(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger: logging.Logger
) -> ModelTrainer:
    """
    Treina todos os modelos.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        
    Returns:
        ModelTrainer com modelos treinados
    """
    logger.info("=" * 70)
    logger.info("ETAPA 4: TREINAMENTO DE MODELOS")
    logger.info("=" * 70)
    
    trainer = ModelTrainer()
    
    # Treinar todos os modelos
    trainer.treinar_todos_modelos(X_train, y_train)
    
    # Salvar modelos
    logger.info("Salvando modelos...")
    trainer.salvar_todos_modelos()
    
    logger.info("Treinamento concluido!")
    
    return trainer


@timer
def executar_avaliacao(
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Avalia todos os modelos.
    
    Args:
        trainer: ModelTrainer com modelos
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        DataFrame com metricas
    """
    logger.info("=" * 70)
    logger.info("ETAPA 5: AVALIACAO DE MODELOS")
    logger.info("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Obter predicoes de todos os modelos
    predicoes = trainer.predizer_todos(X_test)
    
    # Gerar relatorio completo
    relatorio = evaluator.gerar_relatorio_avaliacao(y_test, predicoes)
    
    logger.info("Avaliacao concluida!")
    
    return relatorio['metricas']


@timer
def executar_shap(
    trainer: ModelTrainer,
    X_test: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Executa analise SHAP para interpretabilidade.
    
    Args:
        trainer: ModelTrainer com modelos
        X_test: Features de teste
    """
    logger.info("=" * 70)
    logger.info("ETAPA 6: ANALISE SHAP (INTERPRETABILIDADE)")
    logger.info("=" * 70)
    
    analyzer = SHAPAnalyzer(sample_size=1000)
    
    # Usar XGBoost como modelo principal
    modelo = trainer.models.get('xgboost')
    
    if modelo is None:
        logger.warning("Modelo XGBoost nao encontrado. Pulando SHAP.")
        return
    
    # Gerar relatorio SHAP
    logger.info("Gerando analise SHAP para XGBoost...")
    analyzer.gerar_relatorio_completo(modelo, X_test, tipo_modelo='tree')
    
    logger.info("Analise SHAP concluida!")


def main():
    """Funcao principal do pipeline."""
    # Configurar argumentos
    args = configurar_argumentos()
    
    # Configurar logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("PIPELINE DE ANALISE DE ALFABETIZACAO")
    logger.info("=" * 70)
    logger.info(f"Etapa selecionada: {args.etapa}")
    
    try:
        # ETL
        if args.etapa in ['etl', 'all']:
            df = executar_etl(logger)
            df.to_parquet(DATA_DIR / 'gold' / 'dados_processados.parquet', index=False)
        else:
            # Carregar dados existentes
            arquivo = DATA_DIR / 'gold' / 'dados_processados.parquet'
            if arquivo.exists():
                df = pd.read_parquet(arquivo)
                logger.info(f"Dados carregados: {df.shape}")
            else:
                logger.error("Arquivo de dados nao encontrado. Execute ETL primeiro.")
                return
        
        # EDA
        if args.etapa in ['eda', 'all'] and not args.skip_eda:
            executar_eda(df, logger)
        
        # Feature Engineering
        if args.etapa in ['features', 'train', 'eval', 'shap', 'all']:
            X_train, X_test, y_train, y_test = executar_feature_engineering(df, logger)
        
        # Treinamento
        if args.etapa in ['train', 'eval', 'shap', 'all']:
            trainer = executar_treinamento(X_train, y_train, logger)
        
        # Avaliacao
        if args.etapa in ['eval', 'all']:
            df_metricas = executar_avaliacao(trainer, X_test, y_test, logger)
            
            # Mostrar melhor modelo
            melhor_modelo = df_metricas.iloc[0]
            logger.info(f"\nMelhor modelo: {melhor_modelo['modelo']}")
            logger.info(f"F1-Score: {melhor_modelo['f1_score']:.4f}")
            logger.info(f"ROC-AUC: {melhor_modelo['roc_auc']:.4f}")
        
        # SHAP
        if args.etapa in ['shap', 'all'] and not args.skip_shap:
            executar_shap(trainer, X_test, logger)
        
        logger.info("=" * 70)
        logger.info("PIPELINE CONCLUIDO COM SUCESSO!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Erro durante execucao: {str(e)}")
        raise


if __name__ == "__main__":
    main()
