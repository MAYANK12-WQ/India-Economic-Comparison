"""
Data Skepticism Engine - Core implementation.

Never trust any single data source. Cross-validate everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment."""
    indicator: str
    source: str
    overall_score: float  # 0-10
    methodology_score: float
    consistency_score: float
    timeliness_score: float
    revision_score: float
    political_independence_score: float
    biases_detected: List[str]
    warnings: List[str]
    recommendations: List[str]
    alternative_estimates: Dict[str, float]
    confidence_interval: Tuple[float, float]


@dataclass
class MethodologyChange:
    """Record of a methodology change."""
    date: datetime
    description: str
    impact: str
    backward_comparability: bool
    adjustment_factor: Optional[float] = None


@dataclass
class DataSourceProfile:
    """Profile of a data source."""
    name: str
    institution: str
    methodology_transparency: float  # 0-10
    revision_frequency: str
    political_independence: float  # 0-10
    historical_accuracy: float  # 0-10
    coverage_completeness: float  # 0-10
    methodology_changes: List[MethodologyChange] = field(default_factory=list)


class DataSkepticismEngine:
    """
    Main engine for data skepticism and quality assessment.

    Features:
    - Cross-source validation
    - Methodology change detection
    - Revision pattern analysis
    - Anomaly detection
    - Political bias assessment
    - Alternative estimate generation
    """

    def __init__(self):
        self._source_profiles: Dict[str, DataSourceProfile] = {}
        self._methodology_changes: Dict[str, List[MethodologyChange]] = {}
        self._revision_history: Dict[str, pd.DataFrame] = {}

        # Initialize default source profiles
        self._init_source_profiles()

    def _init_source_profiles(self) -> None:
        """Initialize profiles for known data sources."""
        self._source_profiles = {
            "MOSPI": DataSourceProfile(
                name="MOSPI",
                institution="Ministry of Statistics and Programme Implementation",
                methodology_transparency=7.0,
                revision_frequency="quarterly",
                political_independence=6.0,
                historical_accuracy=8.0,
                coverage_completeness=9.0,
                methodology_changes=[
                    MethodologyChange(
                        date=datetime(2015, 1, 1),
                        description="GDP base year changed from 2004-05 to 2011-12",
                        impact="Significant - growth rates revised upward by ~1.5%",
                        backward_comparability=False,
                        adjustment_factor=1.015,
                    ),
                    MethodologyChange(
                        date=datetime(2015, 1, 1),
                        description="MCA21 database integrated for corporate data",
                        impact="Better coverage of formal sector",
                        backward_comparability=False,
                    ),
                ],
            ),
            "RBI": DataSourceProfile(
                name="RBI",
                institution="Reserve Bank of India",
                methodology_transparency=9.0,
                revision_frequency="monthly",
                political_independence=8.0,
                historical_accuracy=9.0,
                coverage_completeness=9.0,
            ),
            "CMIE": DataSourceProfile(
                name="CMIE",
                institution="Centre for Monitoring Indian Economy",
                methodology_transparency=8.0,
                revision_frequency="weekly",
                political_independence=7.0,
                historical_accuracy=7.0,
                coverage_completeness=6.0,
            ),
            "World Bank": DataSourceProfile(
                name="World Bank",
                institution="World Bank Group",
                methodology_transparency=9.0,
                revision_frequency="annual",
                political_independence=9.0,
                historical_accuracy=9.0,
                coverage_completeness=8.0,
            ),
            "IMF": DataSourceProfile(
                name="IMF",
                institution="International Monetary Fund",
                methodology_transparency=9.0,
                revision_frequency="semi-annual",
                political_independence=9.0,
                historical_accuracy=8.0,
                coverage_completeness=8.0,
            ),
        }

    def assess_data_quality(
        self,
        data: pd.DataFrame,
        indicator: str,
        source: str,
    ) -> DataQualityReport:
        """
        Comprehensive data quality assessment.

        Returns:
            DataQualityReport with detailed quality metrics
        """
        biases = []
        warnings = []
        recommendations = []

        # Get source profile
        profile = self._source_profiles.get(source)
        if profile is None:
            warnings.append(f"Unknown source: {source} - using default profile")
            profile = DataSourceProfile(
                name=source,
                institution="Unknown",
                methodology_transparency=5.0,
                revision_frequency="unknown",
                political_independence=5.0,
                historical_accuracy=5.0,
                coverage_completeness=5.0,
            )

        # Check for methodology changes
        methodology_score = self._assess_methodology(data, indicator, profile, biases)

        # Check data consistency
        consistency_score = self._assess_consistency(data, indicator, warnings)

        # Check timeliness
        timeliness_score = self._assess_timeliness(data, indicator)

        # Check revision patterns
        revision_score = self._assess_revisions(indicator, source, biases)

        # Generate alternative estimates
        alternatives = self._generate_alternatives(data, indicator)

        # Calculate confidence interval
        ci = self._calculate_confidence_interval(data, indicator)

        # Calculate overall score
        overall_score = (
            methodology_score * 0.25 +
            consistency_score * 0.25 +
            timeliness_score * 0.15 +
            revision_score * 0.15 +
            profile.political_independence * 0.20
        )

        # Add recommendations
        if methodology_score < 7:
            recommendations.append(
                "Cross-validate with international sources due to methodology concerns"
            )
        if consistency_score < 7:
            recommendations.append(
                "Data shows inconsistencies - use multiple estimation methods"
            )

        return DataQualityReport(
            indicator=indicator,
            source=source,
            overall_score=overall_score,
            methodology_score=methodology_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            revision_score=revision_score,
            political_independence_score=profile.political_independence,
            biases_detected=biases,
            warnings=warnings,
            recommendations=recommendations,
            alternative_estimates=alternatives,
            confidence_interval=ci,
        )

    def _assess_methodology(
        self,
        data: pd.DataFrame,
        indicator: str,
        profile: DataSourceProfile,
        biases: List[str],
    ) -> float:
        """Assess methodology quality and detect changes."""
        score = profile.methodology_transparency

        # Check for methodology changes during the data period
        data_start = data.index.min()
        data_end = data.index.max()

        for change in profile.methodology_changes:
            if data_start <= change.date <= data_end:
                biases.append(
                    f"Methodology changed on {change.date.date()}: {change.description}"
                )
                if not change.backward_comparability:
                    score -= 1.0
                    biases.append(
                        "Data before and after methodology change not directly comparable"
                    )

        return max(0, min(10, score))

    def _assess_consistency(
        self,
        data: pd.DataFrame,
        indicator: str,
        warnings: List[str],
    ) -> float:
        """Assess internal consistency of data."""
        score = 10.0
        values = data["value"].dropna()

        if len(values) < 4:
            warnings.append("Insufficient data points for consistency check")
            return 5.0

        # Check for suspiciously smooth data
        smoothness = self._calculate_smoothness(values)
        if smoothness > 0.95:
            warnings.append("Data appears suspiciously smooth - may be interpolated")
            score -= 1.5

        # Check for outliers using IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)).sum()

        if outliers > len(values) * 0.1:
            warnings.append(f"High outlier rate: {outliers}/{len(values)}")
            score -= 1.0

        # Check for structural breaks
        has_break, break_date = self._detect_structural_break(values)
        if has_break:
            warnings.append(f"Potential structural break detected around {break_date}")
            score -= 0.5

        return max(0, min(10, score))

    def _calculate_smoothness(self, values: pd.Series) -> float:
        """Calculate how smooth a series is (0=rough, 1=perfectly smooth)."""
        if len(values) < 3:
            return 0.5

        # Calculate first differences
        diffs = values.diff().dropna()

        # Calculate second differences (acceleration)
        second_diffs = diffs.diff().dropna()

        # Smoothness = inverse of second difference variance normalized
        if second_diffs.std() == 0:
            return 1.0

        roughness = second_diffs.std() / values.std()
        smoothness = 1 / (1 + roughness)

        return smoothness

    def _detect_structural_break(
        self,
        values: pd.Series,
    ) -> Tuple[bool, Optional[datetime]]:
        """Detect structural breaks using Chow test approximation."""
        if len(values) < 10:
            return False, None

        # Simple method: look for significant mean shift
        best_break = None
        max_t_stat = 0

        for i in range(len(values) // 4, 3 * len(values) // 4):
            before = values.iloc[:i]
            after = values.iloc[i:]

            # Welch's t-test for different means
            t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)

            if abs(t_stat) > max_t_stat:
                max_t_stat = abs(t_stat)
                best_break = values.index[i]

        # Consider it a break if t-stat > 3
        has_break = max_t_stat > 3

        return has_break, best_break if has_break else None

    def _assess_timeliness(
        self,
        data: pd.DataFrame,
        indicator: str,
    ) -> float:
        """Assess data timeliness."""
        if len(data) == 0:
            return 0.0

        latest_date = data.index.max()
        today = datetime.now()

        # Calculate lag in days
        lag = (today - latest_date).days

        # Score based on lag
        if lag < 30:
            return 10.0
        elif lag < 60:
            return 9.0
        elif lag < 90:
            return 8.0
        elif lag < 180:
            return 6.0
        elif lag < 365:
            return 4.0
        else:
            return 2.0

    def _assess_revisions(
        self,
        indicator: str,
        source: str,
        biases: List[str],
    ) -> float:
        """Assess revision patterns."""
        if indicator not in self._revision_history:
            return 7.0  # Default if no revision history

        revisions = self._revision_history[indicator]

        # Check for systematic revision bias
        if len(revisions) > 0:
            avg_revision = revisions["revision"].mean()

            if abs(avg_revision) > 0.5:
                direction = "upward" if avg_revision > 0 else "downward"
                biases.append(
                    f"Systematic {direction} revision bias detected: avg {avg_revision:.2f}%"
                )
                return 5.0

        return 8.0

    def _generate_alternatives(
        self,
        data: pd.DataFrame,
        indicator: str,
    ) -> Dict[str, float]:
        """Generate alternative estimates using different methods."""
        values = data["value"].dropna()
        if len(values) == 0:
            return {}

        latest = values.iloc[-1]

        return {
            "point_estimate": latest,
            "rolling_avg_4q": values.tail(4).mean() if len(values) >= 4 else latest,
            "trimmed_mean": stats.trim_mean(values, 0.1),
            "median": values.median(),
            "exponential_smoothed": self._exp_smooth(values),
        }

    def _exp_smooth(self, values: pd.Series, alpha: float = 0.3) -> float:
        """Exponential smoothing."""
        result = values.iloc[0]
        for v in values.iloc[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    def _calculate_confidence_interval(
        self,
        data: pd.DataFrame,
        indicator: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for latest estimate."""
        values = data["value"].dropna()

        if len(values) < 4:
            return (values.mean(), values.mean())

        # Use t-distribution for small samples
        mean = values.mean()
        se = stats.sem(values)
        ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=se)

        return ci

    def cross_validate(
        self,
        indicator: str,
        sources: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Cross-validate an indicator across multiple sources.

        Returns:
            Dictionary with validation results
        """
        if len(sources) < 2:
            return {
                "validated": False,
                "reason": "Need at least 2 sources for cross-validation",
            }

        results = {
            "validated": True,
            "sources": list(sources.keys()),
            "discrepancies": [],
            "consensus_estimate": None,
            "reliability_weighted_estimate": None,
        }

        # Calculate pairwise correlations
        correlations = {}
        for name1, data1 in sources.items():
            for name2, data2 in sources.items():
                if name1 >= name2:
                    continue

                # Align data
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) < 4:
                    continue

                corr = data1.loc[common_idx, "value"].corr(
                    data2.loc[common_idx, "value"]
                )
                correlations[f"{name1}-{name2}"] = corr

                # Check for significant discrepancies
                diff = (
                    data1.loc[common_idx, "value"] -
                    data2.loc[common_idx, "value"]
                ).abs()
                avg_diff = diff.mean()
                avg_value = data1.loc[common_idx, "value"].mean()

                if avg_diff / avg_value > 0.1:  # >10% difference
                    results["discrepancies"].append({
                        "sources": [name1, name2],
                        "avg_difference": avg_diff,
                        "pct_difference": avg_diff / avg_value * 100,
                    })

        results["correlations"] = correlations

        # Calculate consensus estimate (simple average)
        latest_values = []
        for name, data in sources.items():
            if len(data) > 0:
                latest_values.append(data["value"].iloc[-1])

        if latest_values:
            results["consensus_estimate"] = np.mean(latest_values)

            # Reliability-weighted estimate
            weights = []
            for name in sources.keys():
                profile = self._source_profiles.get(name)
                if profile:
                    weights.append(profile.historical_accuracy)
                else:
                    weights.append(5.0)

            weights = np.array(weights) / sum(weights)
            results["reliability_weighted_estimate"] = np.average(
                latest_values, weights=weights
            )

        return results

    def detect_potential_biases(
        self,
        data: pd.DataFrame,
        indicator: str,
        source: str,
    ) -> List[str]:
        """
        Detect potential biases in data.

        Returns:
            List of detected bias descriptions
        """
        biases = []

        # Check for election-year effects
        election_years = [2004, 2009, 2014, 2019, 2024]
        for year in election_years:
            if year in data.index.year:
                year_data = data[data.index.year == year]["value"]
                other_data = data[data.index.year != year]["value"]

                if len(year_data) > 0 and len(other_data) > 0:
                    if year_data.mean() > other_data.mean() * 1.1:
                        biases.append(
                            f"Suspiciously high values in election year {year}"
                        )

        # Check for pre-announcement bumps
        values = data["value"].dropna()
        if len(values) >= 12:
            recent_growth = values.iloc[-4:].mean() - values.iloc[-8:-4].mean()
            historical_growth = values.diff().mean()

            if recent_growth > historical_growth * 2:
                biases.append(
                    "Recent growth significantly exceeds historical trend"
                )

        # Check for smoothing
        smoothness = self._calculate_smoothness(values)
        if smoothness > 0.9:
            biases.append(
                "Data appears artificially smoothed (low volatility)"
            )

        return biases

    def generate_skepticism_report(
        self,
        data: Dict[str, pd.DataFrame],
        sources: Dict[str, str],
    ) -> Dict[str, DataQualityReport]:
        """
        Generate comprehensive skepticism report for all indicators.

        Returns:
            Dictionary of indicator name to DataQualityReport
        """
        reports = {}

        for indicator, df in data.items():
            source = sources.get(indicator, "unknown")
            reports[indicator] = self.assess_data_quality(df, indicator, source)

        return reports
