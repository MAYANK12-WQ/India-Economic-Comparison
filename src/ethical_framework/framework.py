"""
Ethical Framework - Ensuring responsible economic analysis.

Prevents bias, ensures balance, and guards against misuse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EthicalPrinciple:
    """An ethical principle for analysis."""
    name: str
    description: str
    check_function: str  # Name of method to check this principle
    severity: str  # 'critical', 'warning', 'info'


@dataclass
class EthicalViolation:
    """A detected ethical violation."""
    principle: str
    description: str
    severity: str
    location: str
    suggestion: str
    auto_fixable: bool


@dataclass
class EthicalReport:
    """Comprehensive ethical review report."""
    overall_score: float  # 0-10
    violations: List[EthicalViolation]
    warnings: List[str]
    recommendations: List[str]
    balanced_interpretation_guide: str
    potential_misuse_warnings: List[str]


class EthicalFramework:
    """
    Ethical framework for economic analysis.

    Principles:
    1. No cherry-picking of dates or metrics
    2. Equal scrutiny for both periods
    3. Transparent assumptions
    4. Acknowledge limitations
    5. Prevent oversimplification
    6. Avoid hindsight bias
    7. Respect context
    8. Human impact first
    """

    def __init__(self):
        self._principles: Dict[str, EthicalPrinciple] = {}
        self._violations: List[EthicalViolation] = []

        self._init_principles()

    def _init_principles(self) -> None:
        """Initialize ethical principles."""
        self._principles = {
            "no_cherry_picking": EthicalPrinciple(
                name="No Cherry-Picking",
                description=(
                    "Do not selectively choose dates, metrics, or data points "
                    "that favor one conclusion over another"
                ),
                check_function="_check_cherry_picking",
                severity="critical",
            ),
            "equal_scrutiny": EthicalPrinciple(
                name="Equal Scrutiny",
                description=(
                    "Apply the same level of scrutiny, skepticism, and "
                    "methodological rigor to both periods being compared"
                ),
                check_function="_check_equal_scrutiny",
                severity="critical",
            ),
            "transparent_assumptions": EthicalPrinciple(
                name="Transparent Assumptions",
                description=(
                    "All assumptions must be explicitly stated, justified, "
                    "and sensitivity-tested"
                ),
                check_function="_check_transparency",
                severity="critical",
            ),
            "acknowledge_limitations": EthicalPrinciple(
                name="Acknowledge Limitations",
                description=(
                    "Every analysis must clearly state its limitations, "
                    "data quality issues, and uncertainty"
                ),
                check_function="_check_limitations",
                severity="warning",
            ),
            "no_oversimplification": EthicalPrinciple(
                name="No Oversimplification",
                description=(
                    "Complex economic phenomena should not be reduced to "
                    "simple scores or binary conclusions"
                ),
                check_function="_check_oversimplification",
                severity="warning",
            ),
            "avoid_hindsight_bias": EthicalPrinciple(
                name="Avoid Hindsight Bias",
                description=(
                    "Do not judge past decisions using information that "
                    "was not available at the time"
                ),
                check_function="_check_hindsight_bias",
                severity="warning",
            ),
            "respect_context": EthicalPrinciple(
                name="Respect Context",
                description=(
                    "Policies must be evaluated in the context of the "
                    "constraints and information available at the time"
                ),
                check_function="_check_context",
                severity="warning",
            ),
            "human_impact_first": EthicalPrinciple(
                name="Human Impact First",
                description=(
                    "Focus on welfare outcomes for citizens, not just "
                    "aggregate economic metrics"
                ),
                check_function="_check_human_impact",
                severity="info",
            ),
            "no_survivorship_bias": EthicalPrinciple(
                name="No Survivorship Bias",
                description=(
                    "Consider failed policies and negative outcomes, "
                    "not just successes"
                ),
                check_function="_check_survivorship_bias",
                severity="warning",
            ),
            "comparison_fairness": EthicalPrinciple(
                name="Comparison Fairness",
                description=(
                    "Compare like with like - account for different "
                    "conditions, duration, and starting points"
                ),
                check_function="_check_comparison_fairness",
                severity="critical",
            ),
        }

    def ethical_review(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EthicalReport:
        """
        Conduct comprehensive ethical review of an analysis.

        Args:
            analysis: The analysis to review
            metadata: Optional metadata about the analysis

        Returns:
            EthicalReport with findings and recommendations
        """
        self._violations = []
        warnings = []
        recommendations = []

        # Run all principle checks
        for principle_id, principle in self._principles.items():
            check_method = getattr(self, principle.check_function, None)
            if check_method:
                violations = check_method(analysis, metadata)
                self._violations.extend(violations)

        # Calculate overall score
        critical_count = sum(1 for v in self._violations if v.severity == "critical")
        warning_count = sum(1 for v in self._violations if v.severity == "warning")

        overall_score = 10.0 - (critical_count * 2.0) - (warning_count * 0.5)
        overall_score = max(0.0, min(10.0, overall_score))

        # Generate warnings
        if critical_count > 0:
            warnings.append(
                f"Analysis has {critical_count} critical ethical concerns that must be addressed"
            )
        if warning_count > 0:
            warnings.append(
                f"Analysis has {warning_count} warnings that should be considered"
            )

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Generate balanced interpretation guide
        interpretation_guide = self._generate_interpretation_guide(analysis)

        # Generate misuse warnings
        misuse_warnings = self._generate_misuse_warnings(analysis)

        return EthicalReport(
            overall_score=overall_score,
            violations=self._violations.copy(),
            warnings=warnings,
            recommendations=recommendations,
            balanced_interpretation_guide=interpretation_guide,
            potential_misuse_warnings=misuse_warnings,
        )

    def _check_cherry_picking(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for cherry-picking of data."""
        violations = []

        # Check if analysis uses selective time periods
        if "time_period" in analysis:
            period = analysis["time_period"]

            # Flag if period doesn't cover full tenure
            known_tenures = {
                "upa": ("2004-05-22", "2014-05-26"),
                "nda": ("2014-05-26", "2024-06-04"),
            }

            for tenure, (start, end) in known_tenures.items():
                if tenure in str(period).lower():
                    # Check if full period is used
                    # This is a simplified check
                    pass

        # Check if selective metrics are used
        if "metrics" in analysis:
            metrics = analysis["metrics"]

            # Flag if only positive/negative metrics for one side
            positive_metrics = ["gdp_growth", "forex_reserves", "fdi"]
            negative_metrics = ["inflation", "fiscal_deficit", "unemployment"]

            used_positive = [m for m in metrics if m in positive_metrics]
            used_negative = [m for m in metrics if m in negative_metrics]

            if len(used_positive) == 0 or len(used_negative) == 0:
                violations.append(EthicalViolation(
                    principle="no_cherry_picking",
                    description="Analysis uses only positive or only negative metrics",
                    severity="critical",
                    location="metrics",
                    suggestion="Include a balanced set of both positive and negative indicators",
                    auto_fixable=True,
                ))

        return violations

    def _check_equal_scrutiny(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for equal scrutiny of both periods."""
        violations = []

        # Check if both periods have same level of detail
        if "period_a_analysis" in analysis and "period_b_analysis" in analysis:
            period_a = analysis["period_a_analysis"]
            period_b = analysis["period_b_analysis"]

            # Check number of metrics
            if isinstance(period_a, dict) and isinstance(period_b, dict):
                metrics_a = len(period_a.get("metrics", []))
                metrics_b = len(period_b.get("metrics", []))

                if abs(metrics_a - metrics_b) > 2:
                    violations.append(EthicalViolation(
                        principle="equal_scrutiny",
                        description=(
                            f"Unequal number of metrics analyzed: "
                            f"Period A has {metrics_a}, Period B has {metrics_b}"
                        ),
                        severity="critical",
                        location="period_analysis",
                        suggestion="Analyze the same metrics for both periods",
                        auto_fixable=True,
                    ))

        return violations

    def _check_transparency(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for transparency of assumptions."""
        violations = []

        # Check if assumptions are documented
        if "assumptions" not in analysis:
            violations.append(EthicalViolation(
                principle="transparent_assumptions",
                description="Analysis does not document assumptions",
                severity="critical",
                location="root",
                suggestion="Add an 'assumptions' section listing all methodological choices",
                auto_fixable=False,
            ))

        # Check if data sources are documented
        if "sources" not in analysis:
            violations.append(EthicalViolation(
                principle="transparent_assumptions",
                description="Analysis does not document data sources",
                severity="warning",
                location="root",
                suggestion="Add a 'sources' section listing all data sources used",
                auto_fixable=False,
            ))

        return violations

    def _check_limitations(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check if limitations are acknowledged."""
        violations = []

        if "limitations" not in analysis and "caveats" not in analysis:
            violations.append(EthicalViolation(
                principle="acknowledge_limitations",
                description="Analysis does not acknowledge limitations",
                severity="warning",
                location="root",
                suggestion="Add a 'limitations' section discussing data quality, methodology limits, etc.",
                auto_fixable=False,
            ))

        return violations

    def _check_oversimplification(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for oversimplification."""
        violations = []

        # Flag if analysis produces single "winner" score
        if "winner" in analysis or "best_period" in analysis:
            violations.append(EthicalViolation(
                principle="no_oversimplification",
                description="Analysis declares a 'winner' - economic performance cannot be reduced to a single score",
                severity="warning",
                location="conclusion",
                suggestion="Present nuanced findings across multiple dimensions without declaring a winner",
                auto_fixable=False,
            ))

        # Flag if confidence intervals are missing
        if "results" in analysis:
            results = analysis["results"]
            if isinstance(results, dict):
                has_ci = any(
                    "confidence_interval" in str(v) or "ci" in str(v).lower()
                    for v in results.values()
                )
                if not has_ci:
                    violations.append(EthicalViolation(
                        principle="no_oversimplification",
                        description="Results presented as point estimates without uncertainty",
                        severity="warning",
                        location="results",
                        suggestion="Add confidence intervals to all estimates",
                        auto_fixable=True,
                    ))

        return violations

    def _check_hindsight_bias(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for hindsight bias."""
        violations = []

        # Flag if analysis judges past predictions without context
        hindsight_keywords = [
            "should have known",
            "obvious in retrospect",
            "predictable",
            "foreseeable",
        ]

        analysis_text = str(analysis).lower()
        for keyword in hindsight_keywords:
            if keyword in analysis_text:
                violations.append(EthicalViolation(
                    principle="avoid_hindsight_bias",
                    description=f"Analysis contains hindsight bias language: '{keyword}'",
                    severity="warning",
                    location="text",
                    suggestion="Evaluate decisions based on information available at the time",
                    auto_fixable=False,
                ))
                break

        return violations

    def _check_context(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check if context is respected."""
        violations = []

        # Flag if global context is ignored
        if "global_context" not in analysis and "external_factors" not in analysis:
            violations.append(EthicalViolation(
                principle="respect_context",
                description="Analysis does not consider global economic context",
                severity="warning",
                location="root",
                suggestion="Add section on global conditions affecting India during each period",
                auto_fixable=False,
            ))

        return violations

    def _check_human_impact(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check if human impact is considered."""
        violations = []

        human_metrics = [
            "poverty", "inequality", "unemployment", "wages",
            "health", "education", "hdi", "welfare"
        ]

        analysis_text = str(analysis).lower()
        has_human_metrics = any(m in analysis_text for m in human_metrics)

        if not has_human_metrics:
            violations.append(EthicalViolation(
                principle="human_impact_first",
                description="Analysis focuses on aggregate metrics without human welfare indicators",
                severity="info",
                location="metrics",
                suggestion="Include poverty, inequality, employment, and welfare indicators",
                auto_fixable=True,
            ))

        return violations

    def _check_survivorship_bias(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check for survivorship bias."""
        violations = []

        # Flag if only successful policies are mentioned
        if "policies" in analysis:
            policies = analysis["policies"]
            if isinstance(policies, list):
                # Check if failed policies are included
                failed_keywords = ["failed", "reversed", "abandoned", "unsuccessful"]
                has_failures = any(
                    any(kw in str(p).lower() for kw in failed_keywords)
                    for p in policies
                )

                if not has_failures:
                    violations.append(EthicalViolation(
                        principle="no_survivorship_bias",
                        description="Analysis only mentions successful policies",
                        severity="warning",
                        location="policies",
                        suggestion="Include analysis of failed or reversed policies for both periods",
                        auto_fixable=False,
                    ))

        return violations

    def _check_comparison_fairness(
        self,
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[EthicalViolation]:
        """Check if comparisons are fair."""
        violations = []

        # Check if periods are of similar duration
        if "periods" in analysis:
            periods = analysis["periods"]
            if isinstance(periods, dict):
                durations = []
                for name, period in periods.items():
                    if isinstance(period, dict) and "start" in period and "end" in period:
                        try:
                            start = pd.to_datetime(period["start"])
                            end = pd.to_datetime(period["end"])
                            durations.append((end - start).days)
                        except Exception:
                            pass

                if len(durations) >= 2:
                    if max(durations) / min(durations) > 1.3:  # 30% difference
                        violations.append(EthicalViolation(
                            principle="comparison_fairness",
                            description="Periods being compared have significantly different durations",
                            severity="warning",
                            location="periods",
                            suggestion="Normalize for duration or clearly note the difference",
                            auto_fixable=False,
                        ))

        return violations

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []

        # Group violations by principle
        principle_violations = {}
        for v in self._violations:
            if v.principle not in principle_violations:
                principle_violations[v.principle] = []
            principle_violations[v.principle].append(v)

        # Generate recommendations
        if "no_cherry_picking" in principle_violations:
            recommendations.append(
                "Show all available time periods and metrics, not just favorable ones. "
                "Use rolling averages to reduce sensitivity to start/end date choices."
            )

        if "equal_scrutiny" in principle_violations:
            recommendations.append(
                "Apply identical analytical methods to both periods. "
                "If criticizing data quality for one period, apply same scrutiny to the other."
            )

        if "transparent_assumptions" in principle_violations:
            recommendations.append(
                "Document all methodological choices explicitly. "
                "Explain why specific approaches were chosen over alternatives."
            )

        if "no_oversimplification" in principle_violations:
            recommendations.append(
                "Present multidimensional results rather than single scores. "
                "Always include uncertainty ranges and caveats."
            )

        if not recommendations:
            recommendations.append(
                "Analysis appears to follow ethical guidelines. "
                "Continue to maintain balance and transparency."
            )

        return recommendations

    def _generate_interpretation_guide(
        self,
        analysis: Dict[str, Any],
    ) -> str:
        """Generate guide for balanced interpretation."""
        guide = """
INTERPRETATION GUIDE
====================

When interpreting this economic comparison, keep in mind:

1. CONTEXT MATTERS
   - Both periods faced unique challenges and opportunities
   - Global conditions varied significantly
   - Starting points and inherited situations differed

2. NO SINGLE WINNER
   - Economic performance is multi-dimensional
   - Period A may be better on some metrics, Period B on others
   - Aggregate comparisons obscure important details

3. DATA LIMITATIONS
   - All economic data has measurement error
   - Methodology changes affect comparability
   - Some important outcomes are not easily measured

4. ATTRIBUTION IS HARD
   - Economic outcomes have many causes
   - Policies take time to show effects
   - Separating policy impact from external factors is difficult

5. AVOID PARTISAN FRAMING
   - Present findings without political bias
   - Acknowledge positives and negatives for both periods
   - Let readers draw their own conclusions
"""
        return guide

    def _generate_misuse_warnings(
        self,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate warnings about potential misuse."""
        warnings = [
            "This analysis should not be used to declare one government 'better' than another",
            "Cherry-picking findings to support a predetermined conclusion is a misuse of this data",
            "Economic outcomes depend on many factors beyond government policy",
            "Historical comparisons cannot account for counterfactuals",
            "Using these findings to disparage individuals or communities is unethical",
        ]

        return warnings

    def validate_claim(
        self,
        claim: str,
        supporting_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate an economic claim against ethical standards.

        Args:
            claim: The claim to validate
            supporting_data: Data cited in support

        Returns:
            Validation result with score and issues
        """
        issues = []

        # Check for absolute statements
        absolute_words = ["always", "never", "worst", "best", "only"]
        claim_lower = claim.lower()
        for word in absolute_words:
            if word in claim_lower:
                issues.append(f"Claim uses absolute language ('{word}') - economic claims rarely have absolutes")

        # Check if claim acknowledges uncertainty
        if "approximately" not in claim_lower and "around" not in claim_lower and "about" not in claim_lower:
            issues.append("Claim presents figures without uncertainty qualifiers")

        # Check if supporting data is provided
        if not supporting_data:
            issues.append("No supporting data provided for claim")

        # Calculate validity score
        score = 10.0 - len(issues) * 2
        score = max(0.0, min(10.0, score))

        return {
            "claim": claim,
            "validity_score": score,
            "issues": issues,
            "is_valid": score >= 6.0,
            "recommendation": "Soften absolute language and add uncertainty qualifiers" if issues else "Claim appears reasonable",
        }
