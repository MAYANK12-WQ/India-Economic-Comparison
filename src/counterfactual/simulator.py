"""
Counterfactual Simulator - What-if scenario analysis.

Simulates alternative economic trajectories under different conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Definition of a counterfactual scenario."""
    name: str
    description: str
    modifications: Dict[str, Any]
    period: Tuple[datetime, datetime]
    assumptions: List[str]


@dataclass
class SimulationResult:
    """Result of a counterfactual simulation."""
    scenario: Scenario
    actual_trajectory: pd.DataFrame
    counterfactual_trajectory: pd.DataFrame
    difference: pd.DataFrame
    cumulative_impact: float
    annual_impact: float
    confidence_interval: Tuple[float, float]
    key_insights: List[str]


class CounterfactualSimulator:
    """
    Simulates "what if" scenarios for economic analysis.

    Predefined scenarios:
    - What if 2008 crisis happened in 2014?
    - What if demonetization never happened?
    - What if GST was implemented in 2009?
    - What if COVID hit in 2009?
    - What if global conditions were equal across periods?
    - What if demographic structure was the same?

    Users can also create custom scenarios.
    """

    def __init__(self):
        self._scenarios: Dict[str, Scenario] = {}
        self._economic_model: Optional[Callable] = None
        self._elasticities: Dict[str, Dict[str, float]] = {}

        # Initialize default scenarios
        self._init_default_scenarios()
        self._init_elasticities()

    def _init_default_scenarios(self) -> None:
        """Initialize predefined scenarios."""
        self._scenarios = {
            "2008_in_2014": Scenario(
                name="2008 Crisis in 2014",
                description="Simulate the 2008 financial crisis occurring in 2014 instead",
                modifications={
                    "global_growth_shock": -3.5,
                    "trade_shock": -15.0,
                    "fdi_shock": -40.0,
                    "duration_quarters": 6,
                },
                period=(datetime(2014, 9, 1), datetime(2016, 3, 31)),
                assumptions=[
                    "Similar fiscal space for response",
                    "Similar monetary policy transmission",
                    "Similar trade openness",
                ],
            ),
            "no_demonetization": Scenario(
                name="No Demonetization",
                description="What if demonetization never happened in 2016",
                modifications={
                    "cash_gdp_ratio": "maintain_pre_demo_trend",
                    "informal_sector_shock": 0,
                    "digital_adoption_boost": -0.5,  # Slower digital adoption
                },
                period=(datetime(2016, 11, 1), datetime(2018, 3, 31)),
                assumptions=[
                    "Informal sector continues at pre-demo levels",
                    "Digital adoption grows at previous trend",
                    "No fiscal benefits from formalization",
                ],
            ),
            "early_gst": Scenario(
                name="GST in 2009",
                description="What if GST was implemented in 2009 instead of 2017",
                modifications={
                    "logistics_cost_reduction": -15.0,
                    "tax_compliance_boost": 10.0,
                    "implementation_disruption": -1.0,
                    "advance_years": 8,
                },
                period=(datetime(2009, 7, 1), datetime(2014, 6, 30)),
                assumptions=[
                    "Similar implementation challenges",
                    "Similar compliance improvement trajectory",
                    "States agree earlier",
                ],
            ),
            "covid_in_2009": Scenario(
                name="COVID in 2009",
                description="What if a COVID-like pandemic hit in 2009",
                modifications={
                    "gdp_shock_q1": -24.0,  # Similar to Q1 FY21
                    "gdp_shock_q2": -7.0,
                    "recovery_quarters": 4,
                    "fiscal_stimulus": 3.0,  # % of GDP
                },
                period=(datetime(2009, 3, 1), datetime(2010, 6, 30)),
                assumptions=[
                    "Similar healthcare capacity constraints",
                    "Less digital infrastructure for WFH",
                    "Different fiscal space availability",
                ],
            ),
            "equal_global_conditions": Scenario(
                name="Equal Global Conditions",
                description="Normalize for different global economic conditions",
                modifications={
                    "global_growth_adjustment": "equalize",
                    "commodity_price_adjustment": "equalize",
                    "trade_environment_adjustment": "equalize",
                },
                period=(datetime(2004, 1, 1), datetime(2024, 12, 31)),
                assumptions=[
                    "India's responsiveness to global conditions is constant",
                    "Transmission mechanisms are similar",
                ],
            ),
            "equal_demographics": Scenario(
                name="Equal Demographics",
                description="Adjust for different demographic structures",
                modifications={
                    "working_age_ratio_adjustment": "equalize",
                    "dependency_ratio_adjustment": "equalize",
                },
                period=(datetime(2004, 1, 1), datetime(2024, 12, 31)),
                assumptions=[
                    "Demographic dividend contributes similarly to growth",
                    "Labor market participation rates are similar",
                ],
            ),
        }

    def _init_elasticities(self) -> None:
        """Initialize economic elasticities for simulation."""
        self._elasticities = {
            "gdp_growth_rate": {
                "global_growth": 0.7,  # 1% global growth = 0.7% India growth
                "oil_price": -0.15,  # 10% oil increase = -1.5% growth
                "fdi": 0.05,  # 10% FDI increase = 0.5% growth
                "trade": 0.3,  # Trade openness elasticity
                "infrastructure": 0.2,  # Infrastructure spending elasticity
            },
            "inflation": {
                "oil_price": 0.3,
                "food_price": 0.4,
                "money_supply": 0.2,
                "global_inflation": 0.15,
            },
            "unemployment": {
                "gdp_growth": -0.3,  # Okun's law for India
                "informal_sector": -0.2,  # Informal sector as buffer
            },
        }

    def simulate(
        self,
        scenario_name: str,
        actual_data: Dict[str, pd.DataFrame],
        monte_carlo_runs: int = 1000,
    ) -> SimulationResult:
        """
        Run a counterfactual simulation.

        Args:
            scenario_name: Name of the scenario to simulate
            actual_data: Dictionary of indicator data
            monte_carlo_runs: Number of Monte Carlo iterations

        Returns:
            SimulationResult with counterfactual trajectory
        """
        if scenario_name not in self._scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self._scenarios[scenario_name]

        # Get actual trajectory
        actual = self._get_actual_trajectory(actual_data, scenario.period)

        # Generate counterfactual
        counterfactual = self._generate_counterfactual(
            actual_data, scenario, monte_carlo_runs
        )

        # Calculate difference
        difference = self._calculate_difference(actual, counterfactual)

        # Calculate impacts
        cumulative_impact = difference["value"].sum()
        annual_impact = cumulative_impact / (
            (scenario.period[1] - scenario.period[0]).days / 365
        )

        # Confidence interval from Monte Carlo
        ci = self._calculate_ci(actual_data, scenario, monte_carlo_runs)

        # Generate insights
        insights = self._generate_insights(scenario, actual, counterfactual, difference)

        return SimulationResult(
            scenario=scenario,
            actual_trajectory=actual,
            counterfactual_trajectory=counterfactual,
            difference=difference,
            cumulative_impact=cumulative_impact,
            annual_impact=annual_impact,
            confidence_interval=ci,
            key_insights=insights,
        )

    def _get_actual_trajectory(
        self,
        data: Dict[str, pd.DataFrame],
        period: Tuple[datetime, datetime],
    ) -> pd.DataFrame:
        """Extract actual trajectory for the period."""
        if "gdp_growth_rate" in data:
            df = data["gdp_growth_rate"]
        else:
            df = list(data.values())[0]

        mask = (df.index >= period[0]) & (df.index <= period[1])
        return df.loc[mask].copy()

    def _generate_counterfactual(
        self,
        data: Dict[str, pd.DataFrame],
        scenario: Scenario,
        monte_carlo_runs: int,
    ) -> pd.DataFrame:
        """Generate counterfactual trajectory."""
        actual = self._get_actual_trajectory(data, scenario.period)
        counterfactual = actual.copy()

        mods = scenario.modifications

        # Apply modifications based on scenario type
        if "global_growth_shock" in mods:
            # Apply a shock
            shock = mods["global_growth_shock"]
            duration = mods.get("duration_quarters", 4)

            # V-shaped recovery pattern
            shock_pattern = self._create_shock_pattern(len(counterfactual), shock, duration)
            counterfactual["value"] = counterfactual["value"] + shock_pattern

        elif "cash_gdp_ratio" in mods:
            # No demonetization scenario
            # Estimate what growth would have been without informal sector shock
            informal_shock = 1.5  # Estimated demonetization impact
            recovery_quarters = 4

            for i in range(min(recovery_quarters, len(counterfactual))):
                if i < len(counterfactual):
                    counterfactual.iloc[i, counterfactual.columns.get_loc("value")] += (
                        informal_shock * (1 - i / recovery_quarters)
                    )

        elif "logistics_cost_reduction" in mods:
            # Early GST scenario
            # Add cumulative benefits
            benefit_per_year = 0.3  # Estimated GST benefit to growth
            for i in range(len(counterfactual)):
                years_since_implementation = i / 4  # Quarters to years
                cumulative_benefit = min(benefit_per_year * years_since_implementation, 1.5)
                counterfactual.iloc[i, counterfactual.columns.get_loc("value")] += cumulative_benefit

        elif "gdp_shock_q1" in mods:
            # COVID in different period
            shocks = [
                mods.get("gdp_shock_q1", 0),
                mods.get("gdp_shock_q2", 0),
            ]
            for i, shock in enumerate(shocks):
                if i < len(counterfactual):
                    counterfactual.iloc[i, counterfactual.columns.get_loc("value")] += shock

        elif mods.get("global_growth_adjustment") == "equalize":
            # Normalize for global conditions
            counterfactual = self._normalize_global_conditions(data, scenario.period)

        elif mods.get("working_age_ratio_adjustment") == "equalize":
            # Normalize for demographics
            counterfactual = self._normalize_demographics(data, scenario.period)

        return counterfactual

    def _create_shock_pattern(
        self,
        length: int,
        shock_magnitude: float,
        duration: int,
    ) -> np.ndarray:
        """Create a V-shaped shock pattern."""
        pattern = np.zeros(length)

        # Shock hits hardest at the beginning, then recovers
        for i in range(min(duration, length)):
            # V-shape: deepest at quarter 1-2, then recovery
            if i < duration // 2:
                pattern[i] = shock_magnitude * (1 - i / (duration / 2))
            else:
                pattern[i] = shock_magnitude * (i - duration / 2) / (duration / 2) * 0.3

        return pattern

    def _normalize_global_conditions(
        self,
        data: Dict[str, pd.DataFrame],
        period: Tuple[datetime, datetime],
    ) -> pd.DataFrame:
        """
        Normalize Indian growth for global conditions.

        Adjusts growth to what it would have been under average global conditions.
        """
        actual = self._get_actual_trajectory(data, period)
        counterfactual = actual.copy()

        # Average global growth (simulated)
        avg_global_growth = 3.5

        # Assume we have global growth data, otherwise use estimates
        if "global_growth" in data:
            global_data = data["global_growth"]
            global_mask = (global_data.index >= period[0]) & (global_data.index <= period[1])
            global_actual = global_data.loc[global_mask]
        else:
            # Simulate global growth
            global_actual = pd.DataFrame({
                "value": np.random.normal(3.5, 1.0, len(actual))
            }, index=actual.index)

        # Adjust for difference from average
        elasticity = self._elasticities["gdp_growth_rate"]["global_growth"]
        adjustment = (avg_global_growth - global_actual["value"]) * elasticity
        counterfactual["value"] = counterfactual["value"] + adjustment.values

        return counterfactual

    def _normalize_demographics(
        self,
        data: Dict[str, pd.DataFrame],
        period: Tuple[datetime, datetime],
    ) -> pd.DataFrame:
        """
        Normalize for demographic differences.

        Removes the component of growth attributable to demographic dividend timing.
        """
        actual = self._get_actual_trajectory(data, period)
        counterfactual = actual.copy()

        # Demographic dividend contribution estimate
        # Peaks around 2010s for India
        years = actual.index.year
        demographic_contribution = np.where(
            years < 2010,
            0.3 + (years - 2004) * 0.05,  # Rising
            0.6 - (years - 2010) * 0.02,  # Slowly declining
        )

        # Baseline demographic contribution (average)
        baseline = 0.4

        # Adjust
        adjustment = baseline - demographic_contribution
        counterfactual["value"] = counterfactual["value"] + adjustment

        return counterfactual

    def _calculate_difference(
        self,
        actual: pd.DataFrame,
        counterfactual: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate difference between actual and counterfactual."""
        diff = actual.copy()
        diff["value"] = actual["value"] - counterfactual["value"]
        return diff

    def _calculate_ci(
        self,
        data: Dict[str, pd.DataFrame],
        scenario: Scenario,
        monte_carlo_runs: int,
    ) -> Tuple[float, float]:
        """Calculate confidence interval via Monte Carlo."""
        impacts = []

        for _ in range(monte_carlo_runs):
            # Add noise to the counterfactual
            actual = self._get_actual_trajectory(data, scenario.period)
            noise = np.random.normal(0, actual["value"].std() * 0.1, len(actual))

            counterfactual = self._generate_counterfactual(data, scenario, 0)
            counterfactual["value"] = counterfactual["value"] + noise

            diff = actual["value"].sum() - counterfactual["value"].sum()
            impacts.append(diff)

        return (
            float(np.percentile(impacts, 2.5)),
            float(np.percentile(impacts, 97.5)),
        )

    def _generate_insights(
        self,
        scenario: Scenario,
        actual: pd.DataFrame,
        counterfactual: pd.DataFrame,
        difference: pd.DataFrame,
    ) -> List[str]:
        """Generate key insights from the simulation."""
        insights = []

        avg_diff = difference["value"].mean()
        total_diff = difference["value"].sum()

        if avg_diff > 0:
            insights.append(
                f"Under the '{scenario.name}' scenario, actual growth was on average "
                f"{abs(avg_diff):.2f}pp higher than the counterfactual."
            )
        else:
            insights.append(
                f"Under the '{scenario.name}' scenario, actual growth was on average "
                f"{abs(avg_diff):.2f}pp lower than the counterfactual."
            )

        # Peak difference
        peak_idx = difference["value"].abs().idxmax()
        peak_diff = difference.loc[peak_idx, "value"]
        insights.append(
            f"The largest deviation ({peak_diff:.2f}pp) occurred around {peak_idx.strftime('%Y-%m')}."
        )

        # Recovery analysis
        if len(difference) > 4:
            first_half = difference["value"].iloc[:len(difference)//2].mean()
            second_half = difference["value"].iloc[len(difference)//2:].mean()

            if abs(second_half) < abs(first_half) * 0.5:
                insights.append("The counterfactual impact was largely temporary with significant recovery.")
            else:
                insights.append("The counterfactual impact appears to have persistent effects.")

        return insights

    def compare_scenarios(
        self,
        scenario_names: List[str],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, SimulationResult]:
        """
        Run multiple scenarios and compare results.

        Returns:
            Dictionary of scenario name to SimulationResult
        """
        results = {}

        for name in scenario_names:
            try:
                results[name] = self.simulate(name, data)
            except Exception as e:
                logger.warning(f"Failed to simulate {name}: {e}")

        return results

    def create_custom_scenario(
        self,
        name: str,
        description: str,
        modifications: Dict[str, Any],
        period: Tuple[datetime, datetime],
        assumptions: List[str],
    ) -> None:
        """
        Create a custom counterfactual scenario.

        Args:
            name: Unique name for the scenario
            description: Human-readable description
            modifications: Dictionary of economic modifications
            period: Tuple of (start_date, end_date)
            assumptions: List of assumptions for the scenario
        """
        self._scenarios[name] = Scenario(
            name=name,
            description=description,
            modifications=modifications,
            period=period,
            assumptions=assumptions,
        )
        logger.info(f"Created custom scenario: {name}")

    def list_scenarios(self) -> Dict[str, str]:
        """List all available scenarios with descriptions."""
        return {
            name: scenario.description
            for name, scenario in self._scenarios.items()
        }
