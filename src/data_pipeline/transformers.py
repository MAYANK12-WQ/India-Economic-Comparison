"""
Data transformation utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TimeAligner:
    """
    Align time series with different frequencies to a common timeline.

    Handles:
    - Monthly to quarterly conversion
    - Quarterly to annual conversion
    - Different fiscal year conventions
    - Missing data interpolation
    """

    FREQUENCY_ORDER = ["D", "W", "M", "Q", "Y"]

    def __init__(
        self,
        target_frequency: str = "Q",
        fiscal_year_start: int = 4,  # April
        interpolation_method: str = "linear",
    ):
        self.target_frequency = target_frequency
        self.fiscal_year_start = fiscal_year_start
        self.interpolation_method = interpolation_method

    def align(
        self,
        series: pd.DataFrame,
        source_frequency: str,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Align a series to the target frequency.

        Args:
            series: DataFrame with datetime index
            source_frequency: Original frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation: Aggregation method ('mean', 'sum', 'last', 'first')

        Returns:
            DataFrame aligned to target frequency
        """
        if source_frequency == self.target_frequency:
            return series

        source_idx = self.FREQUENCY_ORDER.index(source_frequency)
        target_idx = self.FREQUENCY_ORDER.index(self.target_frequency)

        if source_idx < target_idx:
            # Need to aggregate (e.g., monthly to quarterly)
            return self._aggregate(series, aggregation)
        else:
            # Need to interpolate (e.g., annual to quarterly)
            return self._interpolate(series)

    def _aggregate(
        self,
        series: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        """Aggregate to lower frequency."""
        freq_map = {"Q": "Q", "Y": "Y", "M": "M"}
        target_freq = freq_map.get(self.target_frequency, "Q")

        if method == "mean":
            return series.resample(target_freq).mean()
        elif method == "sum":
            return series.resample(target_freq).sum()
        elif method == "last":
            return series.resample(target_freq).last()
        elif method == "first":
            return series.resample(target_freq).first()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _interpolate(
        self,
        series: pd.DataFrame,
    ) -> pd.DataFrame:
        """Interpolate to higher frequency."""
        freq_map = {"Q": "Q", "M": "M", "D": "D"}
        target_freq = freq_map.get(self.target_frequency, "Q")

        # Create new index at target frequency
        new_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=target_freq,
        )

        # Reindex and interpolate
        result = series.reindex(new_index)

        if self.interpolation_method == "linear":
            result = result.interpolate(method="linear")
        elif self.interpolation_method == "cubic":
            result = result.interpolate(method="cubic")
        elif self.interpolation_method == "time":
            result = result.interpolate(method="time")

        return result

    def to_fiscal_year(
        self,
        series: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert calendar year to Indian fiscal year (Apr-Mar)."""
        result = series.copy()

        # Shift index to fiscal year
        result.index = result.index.to_period("Q-MAR")

        return result


class BaseYearSplicer:
    """
    Splice GDP series with different base years.

    India's GDP series underwent major revisions:
    - Old series: Base year 2004-05
    - New series: Base year 2011-12

    This class provides methods to create consistent long-term series.
    """

    def __init__(
        self,
        method: str = "ratio_splicing",
    ):
        self.method = method

    def splice(
        self,
        old_series: pd.DataFrame,
        new_series: pd.DataFrame,
        overlap_period: Tuple[str, str],
    ) -> pd.DataFrame:
        """
        Splice two series with different base years.

        Args:
            old_series: Series with old base year
            new_series: Series with new base year
            overlap_period: Tuple of (start_date, end_date) for overlap

        Returns:
            Spliced continuous series
        """
        overlap_start, overlap_end = overlap_period

        if self.method == "ratio_splicing":
            return self._ratio_splice(old_series, new_series, overlap_start, overlap_end)
        elif self.method == "growth_rate":
            return self._growth_rate_splice(old_series, new_series, overlap_start, overlap_end)
        elif self.method == "interpolation":
            return self._interpolation_splice(old_series, new_series, overlap_start, overlap_end)
        else:
            raise ValueError(f"Unknown splicing method: {self.method}")

    def _ratio_splice(
        self,
        old_series: pd.DataFrame,
        new_series: pd.DataFrame,
        overlap_start: str,
        overlap_end: str,
    ) -> pd.DataFrame:
        """Ratio splicing method."""
        # Get overlap data
        old_overlap = old_series.loc[overlap_start:overlap_end, "value"]
        new_overlap = new_series.loc[overlap_start:overlap_end, "value"]

        # Calculate average ratio
        ratio = (new_overlap / old_overlap).mean()

        # Adjust old series
        adjusted_old = old_series.copy()
        adjusted_old["value"] = adjusted_old["value"] * ratio

        # Combine series
        cutoff = pd.to_datetime(overlap_start)
        old_part = adjusted_old[adjusted_old.index < cutoff]
        new_part = new_series[new_series.index >= cutoff]

        return pd.concat([old_part, new_part])

    def _growth_rate_splice(
        self,
        old_series: pd.DataFrame,
        new_series: pd.DataFrame,
        overlap_start: str,
        overlap_end: str,
    ) -> pd.DataFrame:
        """Growth rate splicing method."""
        # Calculate growth rates from old series
        old_growth = old_series["value"].pct_change()

        # Back-cast using growth rates
        start_value = new_series.loc[overlap_start, "value"]
        back_casted = [start_value]

        # Get old series dates before overlap
        old_dates = old_series[old_series.index < overlap_start].index

        for i in range(len(old_dates) - 1, -1, -1):
            date = old_dates[i]
            if date in old_growth.index:
                growth = old_growth.loc[date]
                if pd.notna(growth):
                    prev_value = back_casted[-1] / (1 + growth)
                    back_casted.append(prev_value)

        back_casted.reverse()

        # Combine
        back_casted_df = pd.DataFrame({
            "value": back_casted[:-1],
        }, index=old_dates)

        return pd.concat([back_casted_df, new_series.loc[overlap_start:]])

    def _interpolation_splice(
        self,
        old_series: pd.DataFrame,
        new_series: pd.DataFrame,
        overlap_start: str,
        overlap_end: str,
    ) -> pd.DataFrame:
        """Interpolation-based splicing."""
        # Calculate ratio at start and end of overlap
        start_ratio = (
            new_series.loc[overlap_start, "value"] /
            old_series.loc[overlap_start, "value"]
        )
        end_ratio = (
            new_series.loc[overlap_end, "value"] /
            old_series.loc[overlap_end, "value"]
        )

        # Linearly interpolate ratio during overlap
        overlap_dates = old_series.loc[overlap_start:overlap_end].index
        n = len(overlap_dates)
        ratios = np.linspace(start_ratio, end_ratio, n)

        # Adjust old series with interpolated ratios
        adjusted_old = old_series.copy()

        # Before overlap: use start ratio
        mask_before = adjusted_old.index < overlap_start
        adjusted_old.loc[mask_before, "value"] *= start_ratio

        # During overlap: use interpolated ratios
        for i, date in enumerate(overlap_dates):
            adjusted_old.loc[date, "value"] *= ratios[i]

        # After overlap: use new series
        cutoff = pd.to_datetime(overlap_end)
        old_part = adjusted_old[adjusted_old.index <= cutoff]
        new_part = new_series[new_series.index > cutoff]

        return pd.concat([old_part, new_part])


class SeasonalAdjuster:
    """
    Seasonal adjustment for economic time series.

    Supports:
    - X-13ARIMA-SEATS (via statsmodels)
    - Simple moving average
    - STL decomposition
    """

    def __init__(
        self,
        method: str = "stl",
        period: int = 4,  # Quarterly
    ):
        self.method = method
        self.period = period

    def adjust(
        self,
        series: pd.DataFrame,
        column: str = "value",
    ) -> pd.DataFrame:
        """
        Apply seasonal adjustment.

        Args:
            series: DataFrame with time series
            column: Column to adjust

        Returns:
            DataFrame with seasonally adjusted values
        """
        result = series.copy()
        values = series[column].values

        if self.method == "stl":
            adjusted = self._stl_adjust(values)
        elif self.method == "moving_average":
            adjusted = self._moving_average_adjust(values)
        elif self.method == "x13":
            adjusted = self._x13_adjust(series[column])
        else:
            raise ValueError(f"Unknown adjustment method: {self.method}")

        result[f"{column}_sa"] = adjusted
        return result

    def _stl_adjust(self, values: np.ndarray) -> np.ndarray:
        """STL decomposition-based adjustment."""
        from statsmodels.tsa.seasonal import STL

        # Handle missing values
        mask = ~np.isnan(values)
        if mask.sum() < 2 * self.period:
            return values

        try:
            stl = STL(values[mask], period=self.period, robust=True)
            result = stl.fit()

            # Return trend + residual (removing seasonal)
            adjusted = np.full_like(values, np.nan)
            adjusted[mask] = result.trend + result.resid
            return adjusted
        except Exception as e:
            logger.warning(f"STL adjustment failed: {e}")
            return values

    def _moving_average_adjust(self, values: np.ndarray) -> np.ndarray:
        """Simple moving average adjustment."""
        # Calculate centered moving average
        ma = pd.Series(values).rolling(
            window=self.period, center=True
        ).mean().values

        # Calculate seasonal factors
        seasonal = values / ma

        # Average seasonal factors
        seasonal_means = []
        for i in range(self.period):
            season_values = seasonal[i::self.period]
            seasonal_means.append(np.nanmean(season_values))

        # Normalize seasonal factors
        seasonal_means = np.array(seasonal_means)
        seasonal_means /= seasonal_means.mean()

        # Tile seasonal factors
        n_tiles = len(values) // self.period + 1
        full_seasonal = np.tile(seasonal_means, n_tiles)[:len(values)]

        return values / full_seasonal

    def _x13_adjust(self, series: pd.Series) -> np.ndarray:
        """X-13ARIMA-SEATS adjustment (placeholder)."""
        # In production, use statsmodels.tsa.x13 or external X-13 binary
        logger.warning("X-13 not implemented, falling back to STL")
        return self._stl_adjust(series.values)


class Normalizer:
    """
    Normalize economic indicators for fair comparison.

    Methods:
    - Population adjustment
    - PPP adjustment
    - Demographic adjustment
    - Global condition adjustment
    """

    def __init__(self):
        self.population_data: Optional[pd.DataFrame] = None
        self.ppp_factors: Optional[pd.DataFrame] = None
        self.demographic_data: Optional[pd.DataFrame] = None

    def set_population_data(self, data: pd.DataFrame) -> None:
        """Set population data for per-capita calculations."""
        self.population_data = data

    def set_ppp_factors(self, data: pd.DataFrame) -> None:
        """Set PPP conversion factors."""
        self.ppp_factors = data

    def set_demographic_data(self, data: pd.DataFrame) -> None:
        """Set demographic data for age structure adjustments."""
        self.demographic_data = data

    def per_capita(
        self,
        series: pd.DataFrame,
        value_column: str = "value",
    ) -> pd.DataFrame:
        """Convert to per-capita values."""
        if self.population_data is None:
            raise ValueError("Population data not set")

        result = series.copy()

        # Align population data with series
        pop_aligned = self.population_data.reindex(series.index).interpolate()

        result[f"{value_column}_per_capita"] = (
            result[value_column] / pop_aligned["population"] * 1e6
        )

        return result

    def ppp_adjust(
        self,
        series: pd.DataFrame,
        value_column: str = "value",
    ) -> pd.DataFrame:
        """Convert to PPP-adjusted values."""
        if self.ppp_factors is None:
            raise ValueError("PPP factors not set")

        result = series.copy()

        # Align PPP factors
        ppp_aligned = self.ppp_factors.reindex(series.index).interpolate()

        result[f"{value_column}_ppp"] = (
            result[value_column] * ppp_aligned["ppp_factor"]
        )

        return result

    def demographic_adjust(
        self,
        series: pd.DataFrame,
        value_column: str = "value",
    ) -> pd.DataFrame:
        """
        Adjust for demographic structure changes.

        Removes the component of growth attributable to working-age
        population changes (demographic dividend).
        """
        if self.demographic_data is None:
            raise ValueError("Demographic data not set")

        result = series.copy()

        # Calculate working-age population ratio
        demo_aligned = self.demographic_data.reindex(series.index).interpolate()

        working_age_ratio = (
            demo_aligned["working_age_population"] /
            demo_aligned["total_population"]
        )

        # Baseline working age ratio (use first period as reference)
        baseline_ratio = working_age_ratio.iloc[0]

        # Adjustment factor
        adjustment = baseline_ratio / working_age_ratio

        result[f"{value_column}_demo_adjusted"] = (
            result[value_column] * adjustment
        )

        return result

    def z_score_normalize(
        self,
        series: pd.DataFrame,
        value_column: str = "value",
    ) -> pd.DataFrame:
        """Convert to z-scores for comparison."""
        result = series.copy()

        values = result[value_column]
        mean = values.mean()
        std = values.std()

        result[f"{value_column}_zscore"] = (values - mean) / std

        return result

    def min_max_normalize(
        self,
        series: pd.DataFrame,
        value_column: str = "value",
        feature_range: Tuple[float, float] = (0, 1),
    ) -> pd.DataFrame:
        """Min-max normalization."""
        result = series.copy()

        values = result[value_column]
        min_val = values.min()
        max_val = values.max()

        scaled = (values - min_val) / (max_val - min_val)
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

        result[f"{value_column}_normalized"] = scaled

        return result
