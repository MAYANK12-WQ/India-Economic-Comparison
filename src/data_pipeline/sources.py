"""
Data source implementations for various providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Base class for all data sources."""

    def __init__(
        self,
        name: str,
        url: str,
        reliability_score: float,
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.url = url
        self.reliability_score = reliability_score
        self.api_key = api_key
        self._indicator_map: Dict[str, str] = {}

    @abstractmethod
    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for an indicator."""
        pass

    @abstractmethod
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        pass

    def _standardize_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Standardize DataFrame format."""
        result = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        })
        result.set_index("date", inplace=True)
        result.sort_index(inplace=True)
        return result


class MOSPIDataSource(BaseDataSource):
    """
    Ministry of Statistics and Programme Implementation data source.

    Provides:
    - GDP and GVA data
    - Industrial Production Index (IIP)
    - Consumer Price Index (CPI)
    - Wholesale Price Index (WPI)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="MOSPI",
            url="https://mospi.gov.in",
            reliability_score=8.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "gdp_growth_rate": "GDP_GROWTH",
            "gdp_current_prices": "GDP_CURRENT",
            "gdp_constant_prices": "GDP_CONSTANT",
            "gva_agriculture": "GVA_AGRI",
            "gva_industry": "GVA_IND",
            "gva_services": "GVA_SERV",
            "iip": "IIP_GENERAL",
            "cpi_combined": "CPI_COMBINED",
            "wpi": "WPI_ALL",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch MOSPI data.

        Note: In production, this would make actual API calls.
        Here we provide a realistic mock implementation.
        """
        if indicator not in self._indicator_map:
            logger.warning(f"Indicator {indicator} not available from MOSPI")
            return None

        # Generate realistic mock data for demonstration
        # In production, replace with actual API calls
        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate realistic mock data based on historical patterns."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="Q")

        # Base patterns for different indicators
        patterns = {
            "gdp_growth_rate": {
                "base": 7.0,
                "volatility": 1.5,
                "trend": 0.01,
            },
            "cpi_combined": {
                "base": 6.0,
                "volatility": 2.0,
                "trend": -0.02,
            },
            "iip": {
                "base": 4.0,
                "volatility": 3.0,
                "trend": 0.0,
            },
        }

        pattern = patterns.get(indicator, {"base": 5.0, "volatility": 2.0, "trend": 0.0})

        # Generate values with trend and noise
        n = len(date_range)
        trend = np.arange(n) * pattern["trend"]
        noise = np.random.normal(0, pattern["volatility"], n)
        values = pattern["base"] + trend + noise

        # Add realistic structural breaks
        # 2008 financial crisis
        crisis_idx = (date_range >= "2008-09-01") & (date_range <= "2009-06-30")
        values[crisis_idx] -= 3.0

        # 2016 demonetization (if applicable)
        demo_idx = (date_range >= "2016-11-01") & (date_range <= "2017-03-31")
        values[demo_idx] -= 1.5

        # 2020 COVID (if applicable)
        covid_idx = (date_range >= "2020-04-01") & (date_range <= "2020-09-30")
        values[covid_idx] -= 8.0

        df = pd.DataFrame({
            "date": date_range,
            "value": values,
        })
        df.set_index("date", inplace=True)

        return df

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class RBIDataSource(BaseDataSource):
    """
    Reserve Bank of India data source.

    Provides:
    - Monetary aggregates
    - Interest rates
    - Forex reserves
    - Credit growth
    - Banking statistics
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="RBI",
            url="https://rbi.org.in",
            reliability_score=9.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "repo_rate": "REPO_RATE",
            "reverse_repo_rate": "REVERSE_REPO",
            "crr": "CRR",
            "slr": "SLR",
            "forex_reserves": "FOREX_RES",
            "money_supply_m3": "M3",
            "credit_growth": "CREDIT_GROWTH",
            "deposit_growth": "DEPOSIT_GROWTH",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate realistic RBI mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

        patterns = {
            "repo_rate": {"base": 6.5, "volatility": 0.25, "trend": -0.005},
            "forex_reserves": {"base": 350, "volatility": 10, "trend": 1.5},
            "credit_growth": {"base": 12, "volatility": 2, "trend": -0.05},
        }

        pattern = patterns.get(indicator, {"base": 5.0, "volatility": 1.0, "trend": 0.0})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class CMIEDataSource(BaseDataSource):
    """
    Centre for Monitoring Indian Economy data source.

    Provides:
    - Employment and unemployment data
    - Consumer sentiment
    - Business sentiment
    - Household surveys
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="CMIE",
            url="https://cmie.com",
            reliability_score=7.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "unemployment_rate": "UR_OVERALL",
            "labor_force_participation": "LFPR",
            "employment_rate": "ER",
            "consumer_sentiment": "CSI",
            "business_sentiment": "BSI",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate realistic CMIE mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

        patterns = {
            "unemployment_rate": {"base": 6.5, "volatility": 1.0, "trend": 0.02},
            "labor_force_participation": {"base": 43, "volatility": 1.5, "trend": -0.01},
            "consumer_sentiment": {"base": 100, "volatility": 5, "trend": 0.0},
        }

        pattern = patterns.get(indicator, {"base": 50, "volatility": 5, "trend": 0.0})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        # COVID impact on unemployment
        covid_idx = (date_range >= "2020-04-01") & (date_range <= "2020-06-30")
        if indicator == "unemployment_rate":
            values[covid_idx] += 15

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class WorldBankDataSource(BaseDataSource):
    """
    World Bank data source for international comparisons.

    Provides:
    - GDP PPP data
    - Poverty statistics
    - Development indicators
    - Doing Business rankings
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="WorldBank",
            url="https://data.worldbank.org",
            reliability_score=9.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "gdp_ppp": "NY.GDP.MKTP.PP.KD",
            "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",
            "poverty_headcount": "SI.POV.DDAY",
            "gini_coefficient": "SI.POV.GINI",
            "fdi_net_inflows": "BX.KLT.DINV.WD.GD.ZS",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate World Bank mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="Y")

        patterns = {
            "gdp_ppp": {"base": 5000, "volatility": 100, "trend": 200},
            "poverty_headcount": {"base": 25, "volatility": 1, "trend": -1},
            "gini_coefficient": {"base": 35, "volatility": 0.5, "trend": 0.1},
        }

        pattern = patterns.get(indicator, {"base": 50, "volatility": 5, "trend": 1})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class IMFDataSource(BaseDataSource):
    """
    International Monetary Fund data source.

    Provides:
    - World Economic Outlook projections
    - Fiscal Monitor data
    - Balance of payments
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="IMF",
            url="https://imf.org",
            reliability_score=9.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "weo_gdp_growth": "NGDP_RPCH",
            "weo_inflation": "PCPI",
            "fiscal_balance": "GGXCNL_NGDP",
            "current_account": "BCA_NGDPD",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate IMF mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="Y")

        patterns = {
            "weo_gdp_growth": {"base": 7, "volatility": 1.5, "trend": 0.0},
            "weo_inflation": {"base": 6, "volatility": 1.5, "trend": -0.1},
            "fiscal_balance": {"base": -4, "volatility": 0.5, "trend": -0.1},
        }

        pattern = patterns.get(indicator, {"base": 0, "volatility": 1, "trend": 0})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class SatelliteDataSource(BaseDataSource):
    """
    Satellite data source for alternative economic indicators.

    Provides:
    - Night lights data (economic activity proxy)
    - NDVI (agricultural output proxy)
    - Ship tracking (trade activity)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="Satellite",
            url="https://earthdata.nasa.gov",
            reliability_score=7.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "night_lights_intensity": "VIIRS_DNB",
            "ndvi_index": "MODIS_NDVI",
            "port_activity": "AIS_PORT",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate satellite mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

        patterns = {
            "night_lights_intensity": {"base": 100, "volatility": 5, "trend": 0.5},
            "ndvi_index": {"base": 0.5, "volatility": 0.1, "trend": 0.001},
            "port_activity": {"base": 1000, "volatility": 50, "trend": 5},
        }

        pattern = patterns.get(indicator, {"base": 100, "volatility": 10, "trend": 0.5})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())


class HighFrequencyDataSource(BaseDataSource):
    """
    High-frequency data source for nowcasting.

    Provides:
    - UPI transaction volumes
    - GST collections
    - E-way bills
    - Electricity consumption
    - Railway freight
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="HighFrequency",
            url="https://various",
            reliability_score=8.0,
            api_key=api_key,
        )

        self._indicator_map = {
            "upi_transactions": "UPI_VOL",
            "gst_collections": "GST_COL",
            "e_way_bills": "EWAY_COUNT",
            "electricity_consumption": "ELEC_MU",
            "railway_freight": "RAIL_MT",
        }

    def fetch(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        if indicator not in self._indicator_map:
            return None

        return self._generate_mock_data(indicator, start_date, end_date)

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate high-frequency mock data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")

        patterns = {
            "upi_transactions": {"base": 1000, "volatility": 100, "trend": 50},
            "gst_collections": {"base": 100000, "volatility": 5000, "trend": 1000},
            "e_way_bills": {"base": 50000, "volatility": 3000, "trend": 500},
            "electricity_consumption": {"base": 100000, "volatility": 5000, "trend": 300},
            "railway_freight": {"base": 100, "volatility": 5, "trend": 0.2},
        }

        pattern = patterns.get(indicator, {"base": 1000, "volatility": 100, "trend": 10})

        n = len(date_range)
        values = (
            pattern["base"] +
            np.arange(n) * pattern["trend"] +
            np.random.normal(0, pattern["volatility"], n)
        )

        # Ensure non-negative for volume data
        values = np.maximum(values, 0)

        return pd.DataFrame({
            "date": date_range,
            "value": values,
        }).set_index("date")

    def get_available_indicators(self) -> List[str]:
        return list(self._indicator_map.keys())
