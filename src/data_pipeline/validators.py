"""
Data validation utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


class DataValidator:
    """
    Validates economic data for quality and consistency.

    Checks:
    - Missing values
    - Outliers
    - Structural breaks
    - Data type consistency
    - Range validation
    - Temporal consistency
    """

    def __init__(
        self,
        outlier_threshold: float = 3.0,
        missing_threshold: float = 0.1,
    ):
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold

        # Valid ranges for common indicators
        self._valid_ranges = {
            "gdp_growth_rate": (-30, 30),
            "cpi_inflation": (-10, 30),
            "unemployment_rate": (0, 50),
            "fiscal_deficit": (-5, 15),
            "forex_reserves": (0, 1000),
            "fdi_inflows": (-50, 200),
        }

    def validate(
        self,
        data: pd.DataFrame,
        indicator: str,
        column: str = "value",
    ) -> ValidationResult:
        """
        Comprehensive data validation.

        Args:
            data: DataFrame to validate
            indicator: Indicator name for context
            column: Column to validate

        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        errors = []
        warnings = []
        statistics = {}

        # Check for empty data
        if len(data) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["Data is empty"],
                warnings=[],
                statistics={},
            )

        values = data[column]

        # 1. Check missing values
        missing_rate = values.isna().mean()
        statistics["missing_rate"] = missing_rate

        if missing_rate > self.missing_threshold:
            errors.append(
                f"High missing rate: {missing_rate:.1%} > {self.missing_threshold:.1%}"
            )
        elif missing_rate > 0:
            warnings.append(f"Some missing values: {missing_rate:.1%}")

        # 2. Check for outliers
        clean_values = values.dropna()
        if len(clean_values) > 3:
            z_scores = np.abs(stats.zscore(clean_values))
            outlier_count = (z_scores > self.outlier_threshold).sum()
            outlier_rate = outlier_count / len(clean_values)
            statistics["outlier_count"] = outlier_count
            statistics["outlier_rate"] = outlier_rate

            if outlier_rate > 0.05:
                warnings.append(
                    f"High outlier rate: {outlier_count} outliers ({outlier_rate:.1%})"
                )

        # 3. Check value ranges
        if indicator in self._valid_ranges:
            min_valid, max_valid = self._valid_ranges[indicator]
            out_of_range = (
                (clean_values < min_valid) | (clean_values > max_valid)
            ).sum()

            if out_of_range > 0:
                warnings.append(
                    f"{out_of_range} values outside expected range [{min_valid}, {max_valid}]"
                )

        # 4. Check temporal consistency
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for gaps
            expected_freq = pd.infer_freq(data.index)
            if expected_freq is None:
                warnings.append("Irregular time series frequency detected")

            # Check for duplicates
            if data.index.duplicated().any():
                errors.append("Duplicate timestamps detected")

        # 5. Calculate basic statistics
        statistics.update({
            "count": len(values),
            "mean": clean_values.mean() if len(clean_values) > 0 else None,
            "std": clean_values.std() if len(clean_values) > 0 else None,
            "min": clean_values.min() if len(clean_values) > 0 else None,
            "max": clean_values.max() if len(clean_values) > 0 else None,
            "skewness": stats.skew(clean_values) if len(clean_values) > 3 else None,
            "kurtosis": stats.kurtosis(clean_values) if len(clean_values) > 3 else None,
        })

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )

    def check_stationarity(
        self,
        data: pd.DataFrame,
        column: str = "value",
    ) -> Dict[str, Any]:
        """
        Check if time series is stationary.

        Uses ADF test.
        """
        from statsmodels.tsa.stattools import adfuller

        values = data[column].dropna()

        if len(values) < 20:
            return {
                "is_stationary": None,
                "error": "Insufficient data for stationarity test",
            }

        try:
            result = adfuller(values)
            return {
                "is_stationary": result[1] < 0.05,
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "interpretation": (
                    "Series is stationary" if result[1] < 0.05
                    else "Series is non-stationary (has unit root)"
                ),
            }
        except Exception as e:
            return {
                "is_stationary": None,
                "error": str(e),
            }

    def detect_structural_breaks(
        self,
        data: pd.DataFrame,
        column: str = "value",
        significance_level: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Detect structural breaks in time series.

        Uses multiple window sizes for robustness.
        """
        values = data[column].dropna()
        breaks = []

        if len(values) < 20:
            return breaks

        # Test multiple breakpoints
        for i in range(len(values) // 4, 3 * len(values) // 4):
            before = values.iloc[:i]
            after = values.iloc[i:]

            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)

            if p_value < significance_level:
                # Variance ratio test
                var_ratio = before.var() / after.var() if after.var() > 0 else np.inf

                breaks.append({
                    "index": i,
                    "date": data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "mean_before": before.mean(),
                    "mean_after": after.mean(),
                    "variance_ratio": var_ratio,
                })

        # Keep only significant breaks with sufficient separation
        if breaks:
            # Sort by significance
            breaks.sort(key=lambda x: x["p_value"])
            # Keep top 3 most significant
            breaks = breaks[:3]

        return breaks

    def validate_cross_sectional(
        self,
        data_dict: Dict[str, pd.DataFrame],
        expected_indicators: List[str],
    ) -> ValidationResult:
        """
        Validate cross-sectional data across multiple indicators.
        """
        errors = []
        warnings = []
        statistics = {}

        # Check for missing indicators
        missing = set(expected_indicators) - set(data_dict.keys())
        if missing:
            warnings.append(f"Missing indicators: {missing}")

        # Check for date alignment
        all_dates = set()
        for name, df in data_dict.items():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())

        date_coverage = {}
        for name, df in data_dict.items():
            if isinstance(df.index, pd.DatetimeIndex):
                coverage = len(df.index) / len(all_dates) if all_dates else 0
                date_coverage[name] = coverage

        statistics["date_coverage"] = date_coverage

        # Flag indicators with poor coverage
        poor_coverage = [
            name for name, cov in date_coverage.items()
            if cov < 0.8
        ]
        if poor_coverage:
            warnings.append(f"Poor date coverage for: {poor_coverage}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )
