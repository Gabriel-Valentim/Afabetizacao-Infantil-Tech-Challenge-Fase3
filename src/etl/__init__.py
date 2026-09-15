"""
Modulo ETL - Extract, Transform, Load.
Implementa a arquitetura Medallion: Bronze, Silver, Gold.
"""

from .bronze import BronzeLayer
from .silver import SilverLayer
from .gold import GoldLayer
from .pipeline import ETLPipeline

__all__ = ['BronzeLayer', 'SilverLayer', 'GoldLayer', 'ETLPipeline']
