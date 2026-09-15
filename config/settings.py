"""
Configurações globais do projeto de predição de alfabetização.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DADOS_ANTIGOS = Path(r"c:\Users\brdacg10\Downloads\metas\Tech-Challenge-Fase3\dados\antigos")
DADOS_NOVOS = Path(r"c:\Users\brdacg10\Downloads\metas\Tech-Challenge-Fase3\dados\novos")

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "ind_alfabetizado"


@dataclass
class ColumnMapping:
    """Mapeamento de colunas originais para nomes padronizados."""
    
    colunas_padronizadas: Dict[str, str] = field(default_factory=lambda: {
        'NU_ANO_AVALIACAO': 'ano_avaliacao',
        'CO_UF': 'cod_uf',
        'SG_UF': 'sigla_uf',
        'ID_ALUNO': 'id_aluno',
        'TP_SERIE': 'tipo_serie',
        'ID_ESCOLA': 'id_escola',
        'TP_DEPENDENCIA': 'tipo_dependencia',
        'NO_MUNICIPIO': 'nome_municipio',
        'IN_PRESENCA_LP': 'ind_presenca_lp',
        'IN_PREENCHIMENTO_LP': 'ind_preenchimento_lp',
        'CO_CADERNO_LP': 'cod_caderno_lp',
        'VL_PESO_ALUNO_LP': 'peso_aluno_lp',
        'VL_PROFICIENCIA_LP': 'proficiencia_lp',
        'IN_ALFABETIZADO': 'ind_alfabetizado',
    })
    
    uf_para_regiao: Dict[str, str] = field(default_factory=lambda: {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 
        'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 
        'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    })
    
    dependencia_labels: Dict[int, str] = field(default_factory=lambda: {
        1: 'Federal', 2: 'Estadual', 3: 'Municipal', 4: 'Privada'
    })


@dataclass
class FeatureConfig:
    """Configuração de features para modelagem."""
    
    colunas_excluir: List[str] = field(default_factory=lambda: [
        'id_aluno', 'id_escola', 'id_municipio', 'cod_uf', 'ano_avaliacao',
        'nome_municipio', 'sigla_uf', 'nome_regiao', 'desc_dependencia',
        'ind_alfabetizado',
        'proficiencia_lp',
        'tx_resposta_bloco_1', 'tx_gabarito_bloco_1',
        'tx_resposta_bloco_2', 'tx_gabarito_bloco_2',
        'tx_resposta_bloco_3', 'tx_gabarito_bloco_3',
        'tx_resposta_bloco_4', 'tx_gabarito_bloco_4',
        'co_bloco_1', 'co_bloco_2', 'co_bloco_3', 'co_bloco_4',
        'score_infraestrutura_pct',
    ])
    
    colunas_inse: List[str] = field(default_factory=lambda: [
        'inse', 'quantidade_alunos_inse',
        'percentual_nivel_1', 'percentual_nivel_2', 'percentual_nivel_3', 'percentual_nivel_4',
        'percentual_nivel_5', 'percentual_nivel_6', 'percentual_nivel_7', 'percentual_nivel_8'
    ])
    
    colunas_censo: List[str] = field(default_factory=lambda: [
        'agua_filtrada', 'agua_potavel', 'energia_rede_publica',
        'biblioteca', 'laboratorio_informatica', 'quadra_esportes',
        'sala_leitura', 'parque_infantil', 'patio_coberto',
        'internet', 'banda_larga', 'alimentacao'
    ])
    
    colunas_ideb: List[str] = field(default_factory=lambda: [
        'ideb', 'nota_saeb_matematica', 'nota_saeb_lingua_portuguesa', 'taxa_aprovacao'
    ])


@dataclass
class VisualizationConfig:
    """Configurações de visualização."""
    
    cores: Dict[str, str] = field(default_factory=lambda: {
        'primaria': '#2E86AB',
        'secundaria': '#A23B72',
        'sucesso': '#28A745',
        'perigo': '#DC3545',
        'alerta': '#FFC107',
        'info': '#17A2B8',
        'escuro': '#343A40',
        'claro': '#F8F9FA',
        'alfabetizado': '#2E86AB',
        'nao_alfabetizado': '#E94F37',
        'destaque': '#F39237'
    })
    
    paleta_regioes: List[str] = field(default_factory=lambda: [
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'
    ])
    
    figsize_default: tuple = (12, 8)
    figsize_wide: tuple = (16, 8)
    figsize_tall: tuple = (10, 12)
    dpi: int = 100


@dataclass
class ModelConfig:
    """Configurações dos modelos de ML."""
    
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })
    
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1.42,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    })
    
    lgbm_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    })
    
    lr_params: Dict = field(default_factory=lambda: {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })


COLUMN_MAPPING = ColumnMapping()
FEATURE_CONFIG = FeatureConfig()
VIZ_CONFIG = VisualizationConfig()
MODEL_CONFIG = ModelConfig()
