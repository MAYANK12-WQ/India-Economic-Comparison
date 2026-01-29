"""
Data quality scoring system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """A dimension of data quality."""
    name: str
    score: float  # 0-10
    weight: float  # Importance weight
    notes: str


@dataclass
class QualityScore:
    """Overall quality score with dimensions."""
    indicator: str
    source: str
    overall_score: float
    dimensions: List[QualityDimension]
    grade: str  # 'A', 'B', 'C', 'D', 'F'
    recommendation: str


class QualityScorer:
    """
    Comprehensive quality scoring for economic data.

    Dimensions scored:
    - Methodology transparency
    - Revision frequency
    - Political independence
    - Historical accuracy
    - Coverage completeness
    - Timeliness
    - Consistency with alternatives
    """

    def __init__(self):
        # Predefined scores for known sources
        self._source_profiles = {
            "MOSPI": {
                "methodology_transparency": 7.0,
                "revision_frequency": 7.0,
                "political_independence": 6.0,
                "historical_accuracy": 8.0,
                "coverage_completeness": 9.0,
                "timeliness": 6.0,
            },
            "RBI": {
                "methodology_transparency": 9.0,
                "revision_frequency": 8.0,
                "political_independence": 8.0,
                "historical_accuracy": 9.0,
                "coverage_completeness": 9.0,
                "timeliness": 8.0,
            },
            "CMIE": {
                "methodology_transparency": 8.0,
                "revision_frequency": 9.0,
                "political_independence": 7.0,
                "historical_accuracy": 7.0,
                "coverage_completeness": 6.0,
                "timeliness": 9.0,
            },
            "World Bank": {
                "methodology_transparency": 9.0,
                "revision_frequency": 7.0,
                "political_independence": 9.0,
                "historical_accuracy": 9.0,
                "coverage_completeness": 8.0,
                "timeliness": 5.0,
            },
            "IMF": {
                "methodology_transparency": 9.0,
                "revision_frequency": 8.0,
                "political_independence": 9.0,
                "historical_accuracy": 8.0,
                "coverage_completeness": 8.0,
                "timeliness": 6.0,
            },
        }

        # Weights for dimensions
        self._weights = {
            "methodology_transparency": 0.20,
            "revision_frequency": 0.10,
            "political_independence": 0.20,
            "historical_accuracy": 0.20,
            "coverage_completeness": 0.15,
            "timeliness": 0.15,
        }

    def score_source(
        self,
        source: str,
    ) -> QualityScore:
        """
        Get quality score for a data source.

        Args:
            source: Source name

        Returns:
            QualityScore with dimensions
        """
        if source not in self._source_profiles:
            # Return default moderate score
            return self._default_score(source)

        profile = self._source_profiles[source]
        dimensions = []

        for dim_name, score in profile.items():
            weight = self._weights.get(dim_name, 0.1)
            dimensions.append(QualityDimension(
                name=dim_name.replace("_", " ").title(),
                score=score,
                weight=weight,
                notes=self._get_dimension_note(dim_name, score),
            ))

        # Calculate weighted average
        overall = sum(
            dim.score * dim.weight for dim in dimensions
        ) / sum(dim.weight for dim in dimensions)

        grade = self._score_to_grade(overall)
        recommendation = self._get_recommendation(overall, source)

        return QualityScore(
            indicator="all",
            source=source,
            overall_score=overall,
            dimensions=dimensions,
            grade=grade,
            recommendation=recommendation,
        )

    def score_indicator(
        self,
        indicator: str,
        source: str,
        data: pd.DataFrame,
    ) -> QualityScore:
        """
        Score quality for a specific indicator.

        Args:
            indicator: Indicator name
            source: Source name
            data: The data to score

        Returns:
            QualityScore
        """
        # Start with source score
        source_score = self.score_source(source)

        # Adjust based on data characteristics
        adjustments = self._data_quality_adjustments(data)

        dimensions = source_score.dimensions.copy()

        # Add data-specific dimensions
        dimensions.append(QualityDimension(
            name="Data Completeness",
            score=adjustments["completeness"],
            weight=0.15,
            notes=f"{adjustments['missing_rate']:.1%} missing values",
        ))

        dimensions.append(QualityDimension(
            name="Data Consistency",
            score=adjustments["consistency"],
            weight=0.10,
            notes=f"{adjustments['outlier_rate']:.1%} outliers",
        ))

        # Recalculate overall
        overall = sum(
            dim.score * dim.weight for dim in dimensions
        ) / sum(dim.weight for dim in dimensions)

        grade = self._score_to_grade(overall)
        recommendation = self._get_recommendation(overall, source)

        return QualityScore(
            indicator=indicator,
            source=source,
            overall_score=overall,
            dimensions=dimensions,
            grade=grade,
            recommendation=recommendation,
        )

    def _data_quality_adjustments(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate data-specific quality adjustments."""
        if "value" not in data.columns:
            return {
                "completeness": 5.0,
                "consistency": 5.0,
                "missing_rate": 0.5,
                "outlier_rate": 0.1,
            }

        values = data["value"]

        # Missing rate
        missing_rate = values.isna().mean()
        completeness = 10 * (1 - missing_rate)

        # Outlier rate
        clean = values.dropna()
        if len(clean) > 3:
            from scipy import stats
            z_scores = np.abs(stats.zscore(clean))
            outlier_rate = (z_scores > 3).mean()
        else:
            outlier_rate = 0

        consistency = 10 * (1 - min(outlier_rate * 5, 1))

        return {
            "completeness": completeness,
            "consistency": consistency,
            "missing_rate": missing_rate,
            "outlier_rate": outlier_rate,
        }

    def _default_score(self, source: str) -> QualityScore:
        """Return default score for unknown sources."""
        dimensions = [
            QualityDimension(
                name="Unknown Source",
                score=5.0,
                weight=1.0,
                notes="Source not in database - using default moderate score",
            )
        ]

        return QualityScore(
            indicator="all",
            source=source,
            overall_score=5.0,
            dimensions=dimensions,
            grade="C",
            recommendation="Unknown source - verify independently before use",
        )

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 9.0:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B"
        elif score >= 6.0:
            return "C"
        elif score >= 5.0:
            return "D"
        else:
            return "F"

    def _get_dimension_note(self, dimension: str, score: float) -> str:
        """Get note for a dimension score."""
        notes = {
            "methodology_transparency": {
                "high": "Methodology well documented",
                "medium": "Some methodology details available",
                "low": "Limited methodology documentation",
            },
            "revision_frequency": {
                "high": "Frequent, predictable revisions",
                "medium": "Moderate revision frequency",
                "low": "Infrequent or unpredictable revisions",
            },
            "political_independence": {
                "high": "Operationally independent",
                "medium": "Some government influence possible",
                "low": "Subject to political pressures",
            },
            "historical_accuracy": {
                "high": "Historically reliable",
                "medium": "Generally accurate",
                "low": "History of significant revisions",
            },
            "coverage_completeness": {
                "high": "Comprehensive coverage",
                "medium": "Good coverage with gaps",
                "low": "Limited coverage",
            },
            "timeliness": {
                "high": "Timely releases",
                "medium": "Moderate release lag",
                "low": "Significant release delays",
            },
        }

        level = "high" if score >= 8 else ("medium" if score >= 6 else "low")
        return notes.get(dimension, {}).get(level, "")

    def _get_recommendation(self, score: float, source: str) -> str:
        """Get recommendation based on score."""
        if score >= 8.0:
            return f"{source} is a reliable source. Use with standard caveats."
        elif score >= 6.0:
            return f"{source} is acceptable. Cross-validate important findings."
        elif score >= 4.0:
            return f"{source} has quality concerns. Use cautiously, prefer alternatives."
        else:
            return f"{source} has significant issues. Avoid if alternatives exist."

    def compare_sources(
        self,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Compare quality across multiple sources.

        Returns:
            Comparison with rankings
        """
        scores = {source: self.score_source(source) for source in sources}

        ranking = sorted(
            scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True,
        )

        return {
            "scores": {s: sc.overall_score for s, sc in scores.items()},
            "grades": {s: sc.grade for s, sc in scores.items()},
            "ranking": [s for s, _ in ranking],
            "best_source": ranking[0][0] if ranking else None,
            "recommendation": (
                f"Prefer {ranking[0][0]} (Grade {ranking[0][1].grade}) "
                f"when available"
                if ranking else "No sources to compare"
            ),
        }
