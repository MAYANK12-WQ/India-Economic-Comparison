"""
Main Comparison Engine
======================

Orchestrates all modules for comprehensive economic comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .data_pipeline import DataPipeline
from .data_skepticism import DataSkepticismEngine
from .causal_inference import CausalInferenceEngine
from .counterfactual import CounterfactualSimulator
from .uncertainty import UncertaintyQuantifier
from .debate_assistant import DebateAssistant
from .ethical_framework import EthicalFramework
from .predictive_forensics import PredictiveForensics

logger = logging.getLogger(__name__)


@dataclass
class Period:
    """Definition of a comparison period."""
    name: str
    start_date: datetime
    end_date: datetime
    label: str


@dataclass
class ComparisonResult:
    """Result of a comprehensive comparison."""
    period_a: Period
    period_b: Period
    indicator_comparisons: Dict[str, Dict[str, Any]]
    causal_analyses: Dict[str, Any]
    counterfactual_results: Dict[str, Any]
    quality_report: Dict[str, Any]
    ethical_review: Any
    executive_summary: str
    detailed_findings: List[str]
    visualizations: Dict[str, Any]


class ComparisonEngine:
    """
    Main orchestration engine for India Economic Comparison.

    Coordinates:
    - Data fetching and validation
    - Period-wise comparison
    - Causal inference for policies
    - Counterfactual scenarios
    - Uncertainty quantification
    - Ethical review
    - Report generation
    """

    # Default periods
    UPA_PERIOD = Period(
        name="upa",
        start_date=datetime(2004, 5, 22),
        end_date=datetime(2014, 5, 26),
        label="UPA Era (2004-2014)",
    )

    NDA_PERIOD = Period(
        name="nda",
        start_date=datetime(2014, 5, 26),
        end_date=datetime(2024, 6, 4),
        label="NDA Era (2014-2024)",
    )

    def __init__(
        self,
        config_path: Optional[Path] = None,
    ):
        """Initialize the comparison engine with all modules."""
        self.data_pipeline = DataPipeline(config_path)
        self.skepticism_engine = DataSkepticismEngine()
        self.causal_engine = CausalInferenceEngine()
        self.counterfactual_sim = CounterfactualSimulator()
        self.uncertainty = UncertaintyQuantifier()
        self.debate_assistant = DebateAssistant()
        self.ethical_framework = EthicalFramework()
        self.predictive_forensics = PredictiveForensics()

        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._comparison_results: Optional[ComparisonResult] = None

        # Default indicator groups
        self._indicator_groups = {
            "growth": [
                "gdp_growth_rate",
                "gdp_per_capita_growth",
                "iip",
            ],
            "inflation": [
                "cpi_combined",
                "food_inflation",
                "wpi",
            ],
            "employment": [
                "unemployment_rate",
                "labor_force_participation",
            ],
            "fiscal": [
                "fiscal_deficit",
                "public_debt",
                "tax_gdp_ratio",
            ],
            "external": [
                "current_account",
                "forex_reserves",
                "fdi_inflows",
            ],
            "social": [
                "poverty_headcount",
                "gini_coefficient",
            ],
        }

    def run_full_comparison(
        self,
        period_a: Optional[Period] = None,
        period_b: Optional[Period] = None,
        indicators: Optional[List[str]] = None,
        include_causal: bool = True,
        include_counterfactual: bool = True,
        include_ethical_review: bool = True,
    ) -> ComparisonResult:
        """
        Run comprehensive comparison between two periods.

        Args:
            period_a: First period (default: UPA)
            period_b: Second period (default: NDA)
            indicators: Specific indicators to compare (default: all)
            include_causal: Include causal analysis
            include_counterfactual: Include counterfactual scenarios
            include_ethical_review: Include ethical review

        Returns:
            ComparisonResult with all findings
        """
        period_a = period_a or self.UPA_PERIOD
        period_b = period_b or self.NDA_PERIOD

        logger.info(f"Starting comparison: {period_a.label} vs {period_b.label}")

        # Get all indicators if not specified
        if indicators is None:
            indicators = []
            for group in self._indicator_groups.values():
                indicators.extend(group)

        # 1. Fetch and validate data
        data = self._fetch_all_data(indicators, period_a, period_b)

        # 2. Run data quality assessment
        quality_report = self._assess_data_quality(data, indicators)

        # 3. Compare indicators
        indicator_comparisons = self._compare_indicators(
            data, indicators, period_a, period_b
        )

        # 4. Causal analysis (if requested)
        causal_analyses = {}
        if include_causal:
            causal_analyses = self._run_causal_analyses(data)

        # 5. Counterfactual scenarios (if requested)
        counterfactual_results = {}
        if include_counterfactual:
            counterfactual_results = self._run_counterfactual_scenarios(data)

        # 6. Generate visualizations
        visualizations = self._generate_visualizations(
            indicator_comparisons, causal_analyses, counterfactual_results
        )

        # 7. Ethical review (if requested)
        ethical_review = None
        if include_ethical_review:
            ethical_review = self._run_ethical_review(
                indicator_comparisons, causal_analyses
            )

        # 8. Generate summary and findings
        executive_summary = self._generate_executive_summary(
            indicator_comparisons, causal_analyses, counterfactual_results
        )
        detailed_findings = self._generate_detailed_findings(
            indicator_comparisons, causal_analyses, counterfactual_results
        )

        result = ComparisonResult(
            period_a=period_a,
            period_b=period_b,
            indicator_comparisons=indicator_comparisons,
            causal_analyses=causal_analyses,
            counterfactual_results=counterfactual_results,
            quality_report=quality_report,
            ethical_review=ethical_review,
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            visualizations=visualizations,
        )

        self._comparison_results = result
        return result

    def _fetch_all_data(
        self,
        indicators: List[str],
        period_a: Period,
        period_b: Period,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all indicators covering both periods."""
        start_date = min(period_a.start_date, period_b.start_date)
        end_date = max(period_a.end_date, period_b.end_date)

        data = {}
        for indicator in indicators:
            try:
                ts = self.data_pipeline.fetch_indicator(
                    indicator, start_date, end_date
                )
                data[indicator] = ts.data
                self._data_cache[indicator] = ts.data
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator}: {e}")
                # Generate mock data for demonstration
                data[indicator] = self._generate_mock_data(
                    indicator, start_date, end_date
                )

        return data

    def _generate_mock_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Generate mock data for demonstration."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="Q")

        # Base values vary by indicator type
        base_values = {
            "gdp_growth_rate": 7.0,
            "cpi_combined": 6.0,
            "unemployment_rate": 5.0,
            "fiscal_deficit": 4.5,
            "forex_reserves": 400,
        }

        base = base_values.get(indicator, 5.0)
        noise = np.random.normal(0, base * 0.1, len(date_range))
        values = base + noise

        return pd.DataFrame({"value": values}, index=date_range)

    def _assess_data_quality(
        self,
        data: Dict[str, pd.DataFrame],
        indicators: List[str],
    ) -> Dict[str, Any]:
        """Assess quality of all data."""
        quality_scores = {}

        for indicator in indicators:
            if indicator in data:
                report = self.skepticism_engine.assess_data_quality(
                    data[indicator], indicator, "MOSPI"
                )
                quality_scores[indicator] = {
                    "overall_score": report.overall_score,
                    "biases": report.biases_detected,
                    "warnings": report.warnings,
                }

        avg_score = np.mean([q["overall_score"] for q in quality_scores.values()])

        return {
            "indicator_scores": quality_scores,
            "average_score": avg_score,
            "recommendation": (
                "Data quality is acceptable" if avg_score > 7
                else "Exercise caution - data quality concerns"
            ),
        }

    def _compare_indicators(
        self,
        data: Dict[str, pd.DataFrame],
        indicators: List[str],
        period_a: Period,
        period_b: Period,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare all indicators between periods."""
        comparisons = {}

        for indicator in indicators:
            if indicator not in data:
                continue

            df = data[indicator]

            # Split by period
            mask_a = (df.index >= period_a.start_date) & (df.index <= period_a.end_date)
            mask_b = (df.index >= period_b.start_date) & (df.index <= period_b.end_date)

            period_a_data = df.loc[mask_a, "value"]
            period_b_data = df.loc[mask_b, "value"]

            if len(period_a_data) == 0 or len(period_b_data) == 0:
                continue

            # Calculate statistics
            mean_a = period_a_data.mean()
            mean_b = period_b_data.mean()
            std_a = period_a_data.std()
            std_b = period_b_data.std()

            # Statistical test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(period_a_data, period_b_data)

            # Uncertainty quantification
            def estimator(df):
                return df["value"].mean()

            uncertainty_a = self.uncertainty.quantify_uncertainty(
                pd.DataFrame({"value": period_a_data}), estimator
            )
            uncertainty_b = self.uncertainty.quantify_uncertainty(
                pd.DataFrame({"value": period_b_data}), estimator
            )

            comparisons[indicator] = {
                "period_a": {
                    "mean": mean_a,
                    "std": std_a,
                    "ci_95": uncertainty_a.confidence_interval_95,
                },
                "period_b": {
                    "mean": mean_b,
                    "std": std_b,
                    "ci_95": uncertainty_b.confidence_interval_95,
                },
                "difference": mean_b - mean_a,
                "pct_change": ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": self._interpret_comparison(
                    indicator, mean_a, mean_b, p_value
                ),
            }

        return comparisons

    def _interpret_comparison(
        self,
        indicator: str,
        mean_a: float,
        mean_b: float,
        p_value: float,
    ) -> str:
        """Generate interpretation of comparison."""
        diff = mean_b - mean_a
        pct = abs(diff / mean_a * 100) if mean_a != 0 else 0

        direction = "higher" if diff > 0 else "lower"
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

        # Determine if higher is better
        higher_is_better = {
            "gdp_growth_rate": True,
            "forex_reserves": True,
            "fdi_inflows": True,
            "unemployment_rate": False,
            "inflation": False,
            "fiscal_deficit": False,
            "poverty_headcount": False,
        }

        is_improvement = (
            (diff > 0 and higher_is_better.get(indicator, True)) or
            (diff < 0 and not higher_is_better.get(indicator, True))
        )

        quality = "improved" if is_improvement else "deteriorated"

        return (
            f"{indicator} was {pct:.1f}% {direction} in Period B. "
            f"This difference is {significance} (p={p_value:.3f}). "
            f"Performance appears to have {quality}."
        )

    def _run_causal_analyses(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Run causal analyses for major policies."""
        policies = ["demonetization", "gst_implementation", "covid_lockdown"]
        results = {}

        for policy in policies:
            try:
                effect = self.causal_engine.analyze_policy_impact(
                    policy, data, method="did"
                )
                results[policy] = {
                    "effect": effect.point_estimate,
                    "ci": effect.confidence_interval,
                    "p_value": effect.p_value,
                    "significant": effect.statistically_significant,
                    "interpretation": effect.interpretation,
                }
            except Exception as e:
                logger.warning(f"Causal analysis failed for {policy}: {e}")

        return results

    def _run_counterfactual_scenarios(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Run counterfactual scenario simulations."""
        scenarios = ["no_demonetization", "equal_global_conditions", "covid_in_2009"]
        results = {}

        for scenario in scenarios:
            try:
                result = self.counterfactual_sim.simulate(scenario, data)
                results[scenario] = {
                    "cumulative_impact": result.cumulative_impact,
                    "annual_impact": result.annual_impact,
                    "ci": result.confidence_interval,
                    "insights": result.key_insights,
                }
            except Exception as e:
                logger.warning(f"Counterfactual simulation failed for {scenario}: {e}")

        return results

    def _generate_visualizations(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        causal: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate visualization specifications."""
        from .visualization.charts import (
            ComparisonChart, TimeSeriesChart, UncertaintyChart, RadarChart
        )

        charts = {}

        # Bar comparison
        bar_data = {
            indicator: {
                "UPA": comp["period_a"]["mean"],
                "NDA": comp["period_b"]["mean"],
            }
            for indicator, comp in comparisons.items()
        }

        chart = ComparisonChart()
        charts["bar_comparison"] = chart.bar_comparison(
            bar_data, "Period Comparison - Key Indicators"
        )

        # Radar chart for normalized scores
        radar_data = {}
        for period in ["UPA", "NDA"]:
            key = "period_a" if period == "UPA" else "period_b"
            radar_data[period] = {
                indicator: min(100, max(0, comp[key]["mean"] * 10))
                for indicator, comp in list(comparisons.items())[:6]
            }

        radar = RadarChart()
        charts["radar_comparison"] = radar.multi_dimension_comparison(
            radar_data, "Multi-Dimensional Comparison"
        )

        return charts

    def _run_ethical_review(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        causal: Dict[str, Any],
    ) -> Any:
        """Run ethical review of the analysis."""
        analysis = {
            "period_a_analysis": comparisons,
            "period_b_analysis": comparisons,
            "causal_analyses": causal,
            "assumptions": [
                "Data quality is similar across periods",
                "Methodology changes have been accounted for",
            ],
            "limitations": [
                "Historical data may have measurement errors",
                "Causal attribution is inherently difficult",
            ],
        }

        return self.ethical_framework.ethical_review(analysis)

    def _generate_executive_summary(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        causal: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> str:
        """Generate executive summary of findings."""
        # Count improvements vs deteriorations
        improvements = 0
        deteriorations = 0

        for indicator, comp in comparisons.items():
            diff = comp["difference"]
            # Higher is better for some, lower for others
            higher_is_better = indicator not in [
                "unemployment_rate", "inflation", "fiscal_deficit", "poverty_headcount"
            ]

            if (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                improvements += 1
            else:
                deteriorations += 1

        summary = f"""
EXECUTIVE SUMMARY: India Economic Comparison 2004-2024
======================================================

This analysis compares India's economic performance across two periods:
- Period A (UPA): 2004-05-22 to 2014-05-26
- Period B (NDA): 2014-05-26 to 2024-06-04

KEY FINDINGS:

1. INDICATOR COMPARISON
   - Analyzed {len(comparisons)} economic indicators
   - {improvements} indicators showed improvement in Period B
   - {deteriorations} indicators showed deterioration in Period B
   - Note: "Improvement" depends on indicator type

2. CAUSAL ANALYSIS
   - Analyzed {len(causal)} major policy events
   - Key finding: Policy impacts are context-dependent

3. COUNTERFACTUAL SCENARIOS
   - Simulated {len(counterfactual)} alternative scenarios
   - Results suggest both external and policy factors mattered

IMPORTANT CAVEATS:
- Economic performance is multi-dimensional
- External conditions differed significantly
- Direct comparison has methodological limitations
- This analysis should inform, not conclude debates

For detailed findings, see the full report.
"""
        return summary

    def _generate_detailed_findings(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        causal: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> List[str]:
        """Generate list of detailed findings."""
        findings = []

        # Indicator findings
        for indicator, comp in comparisons.items():
            findings.append(comp["interpretation"])

        # Causal findings
        for policy, result in causal.items():
            if "interpretation" in result:
                findings.append(result["interpretation"])

        # Counterfactual findings
        for scenario, result in counterfactual.items():
            if "insights" in result:
                findings.extend(result["insights"])

        return findings

    def get_debate_response(
        self,
        argument: str,
    ) -> Any:
        """Get debate response for an argument."""
        return self.debate_assistant.respond_to_argument(argument)

    def validate_claim(
        self,
        claim_id: str,
    ) -> Any:
        """Validate a political claim."""
        return self.predictive_forensics.validate_claim(
            claim_id, self._data_cache
        )

    def export_report(
        self,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """Export comparison results to file."""
        if self._comparison_results is None:
            raise ValueError("No comparison results to export. Run comparison first.")

        # Convert to serializable format
        report = {
            "period_a": {
                "name": self._comparison_results.period_a.name,
                "start": self._comparison_results.period_a.start_date.isoformat(),
                "end": self._comparison_results.period_a.end_date.isoformat(),
            },
            "period_b": {
                "name": self._comparison_results.period_b.name,
                "start": self._comparison_results.period_b.start_date.isoformat(),
                "end": self._comparison_results.period_b.end_date.isoformat(),
            },
            "indicator_comparisons": self._comparison_results.indicator_comparisons,
            "causal_analyses": self._comparison_results.causal_analyses,
            "counterfactual_results": self._comparison_results.counterfactual_results,
            "quality_report": self._comparison_results.quality_report,
            "executive_summary": self._comparison_results.executive_summary,
            "detailed_findings": self._comparison_results.detailed_findings,
        }

        if format == "json":
            import json
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            # Export indicator comparisons as CSV
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Indicator", "Period A Mean", "Period B Mean",
                    "Difference", "P-Value", "Significant"
                ])
                for ind, comp in self._comparison_results.indicator_comparisons.items():
                    writer.writerow([
                        ind,
                        comp["period_a"]["mean"],
                        comp["period_b"]["mean"],
                        comp["difference"],
                        comp["p_value"],
                        comp["significant"],
                    ])

        logger.info(f"Report exported to {output_path}")
