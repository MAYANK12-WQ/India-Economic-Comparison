"""
Data Pipeline Module
====================

Handles all data fetching, cleaning, validation, and transformation.
Supports multiple data sources with automatic retry and caching.
"""

from .pipeline import DataPipeline
from .sources import (
    MOSPIDataSource,
    RBIDataSource,
    CMIEDataSource,
    WorldBankDataSource,
    IMFDataSource,
    SatelliteDataSource,
    HighFrequencyDataSource,
)
from .transformers import (
    TimeAligner,
    BaseYearSplicer,
    SeasonalAdjuster,
    Normalizer,
)
from .validators import DataValidator

__all__ = [
    "DataPipeline",
    "MOSPIDataSource",
    "RBIDataSource",
    "CMIEDataSource",
    "WorldBankDataSource",
    "IMFDataSource",
    "SatelliteDataSource",
    "HighFrequencyDataSource",
    "TimeAligner",
    "BaseYearSplicer",
    "SeasonalAdjuster",
    "Normalizer",
    "DataValidator",
]
