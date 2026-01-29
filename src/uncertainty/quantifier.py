"""
Uncertainty Quantifier - Every number has a confidence interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """An estimate with full uncertainty quantification."""
    point_estimate: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_90: Tuple[float, float]
    confidence_interval_80: Tuple[float, float]
    distribution: str  # 'normal', 't', 'bootstrap', 'bayesian'
    method_variation: Dict[str, float]
    assumption_sensitivity: Dict[str, Tuple[float, float]]
    data_quality_adjustment: float
    interpretation: str


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    parameter: str
    baseline_value: Any
    tested_range: List[Any]
    results: List[float]
    sensitivity_coefficient: float
    is_sensitive: bool
    recommendation: str


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for economic estimates.

    Features:
    - Bootstrap confidence intervals
    - Bayesian credible intervals
    - Monte Carlo simulation
    - Sensitivity analysis
    - Multiple estimation method comparison
    - Data quality adjustments
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 10000,
        monte_carlo_simulations: int = 5000,
    ):
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.monte_carlo_simulations = monte_carlo_simulations

    def quantify_uncertainty(
        self,
        data: pd.DataFrame,
        estimator: Callable[[pd.DataFrame], float],
        column: str = "value",
        data_quality_score: float = 1.0,
    ) -> UncertaintyEstimate:
        """
        Comprehensive uncertainty quantification for an estimate.

        Args:
            data: DataFrame with the data
            estimator: Function that produces a point estimate
            column: Column to analyze
            data_quality_score: Quality score (0-1) to adjust intervals

        Returns:
            UncertaintyEstimate with full uncertainty information
        """
        values = data[column].dropna()

        if len(values) < 2:
            raise ValueError("Insufficient data for uncertainty quantification")

        # Point estimate
        point = estimator(data)

        # Bootstrap confidence intervals
        bootstrap_ci, bootstrap_se = self._bootstrap_ci(values, estimator)

        # Method variation
        method_variation = self._estimate_method_variation(values)

        # Assumption sensitivity
        sensitivity = self._assumption_sensitivity(values, point)

        # Data quality adjustment
        adjustment = self._data_quality_adjustment(data_quality_score, bootstrap_se)

        # Adjusted confidence intervals
        adjusted_se = bootstrap_se * (1 + adjustment)
        ci_95 = (
            point - 1.96 * adjusted_se,
            point + 1.96 * adjusted_se,
        )
        ci_90 = (
            point - 1.645 * adjusted_se,
            point + 1.645 * adjusted_se,
        )
        ci_80 = (
            point - 1.28 * adjusted_se,
            point + 1.28 * adjusted_se,
        )

        # Interpretation
        interpretation = self._generate_interpretation(
            point, ci_95, method_variation, sensitivity
        )

        return UncertaintyEstimate(
            point_estimate=point,
            standard_error=adjusted_se,
            confidence_interval_95=ci_95,
            confidence_interval_90=ci_90,
            confidence_interval_80=ci_80,
            distribution="bootstrap",
            method_variation=method_variation,
            assumption_sensitivity=sensitivity,
            data_quality_adjustment=adjustment,
            interpretation=interpretation,
        )

    def _bootstrap_ci(
        self,
        values: pd.Series,
        estimator: Callable,
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_estimates = []

        for _ in range(self.bootstrap_iterations):
            sample = values.sample(frac=1, replace=True)
            df = pd.DataFrame({values.name or "value": sample})
            try:
                estimate = estimator(df)
                bootstrap_estimates.append(estimate)
            except Exception:
                continue

        if not bootstrap_estimates:
            # Fallback to parametric
            se = values.std() / np.sqrt(len(values))
            mean = values.mean()
            return (mean - 1.96 * se, mean + 1.96 * se), se

        bootstrap_estimates = np.array(bootstrap_estimates)
        ci = (
            np.percentile(bootstrap_estimates, 2.5),
            np.percentile(bootstrap_estimates, 97.5),
        )
        se = np.std(bootstrap_estimates)

        return ci, se

    def _estimate_method_variation(
        self,
        values: pd.Series,
    ) -> Dict[str, float]:
        """Estimate variation across different estimation methods."""
        return {
            "mean": float(values.mean()),
            "trimmed_mean_10": float(stats.trim_mean(values, 0.1)),
            "trimmed_mean_20": float(stats.trim_mean(values, 0.2)),
            "median": float(values.median()),
            "winsorized_mean": float(stats.mstats.winsorize(values, limits=[0.1, 0.1]).mean()),
            "exponential_weighted": float(values.ewm(span=len(values)//2).mean().iloc[-1]),
        }

    def _assumption_sensitivity(
        self,
        values: pd.Series,
        point: float,
    ) -> Dict[str, Tuple[float, float]]:
        """Analyze sensitivity to key assumptions."""
        sensitivity = {}

        # Sensitivity to inflation measure
        # Assume ±0.5% variation in inflation adjustment
        inflation_range = 0.5
        sensitivity["inflation_measure"] = (
            point - inflation_range,
            point + inflation_range,
        )

        # Sensitivity to base year
        # Assume ±0.3% variation due to base year choice
        base_year_range = 0.3
        sensitivity["base_year"] = (
            point - base_year_range,
            point + base_year_range,
        )

        # Sensitivity to seasonal adjustment
        seasonal_range = values.std() * 0.1
        sensitivity["seasonal_adjustment"] = (
            point - seasonal_range,
            point + seasonal_range,
        )

        # Sensitivity to outlier treatment
        trimmed = stats.trim_mean(values, 0.1)
        sensitivity["outlier_treatment"] = (
            min(point, trimmed),
            max(point, trimmed),
        )

        return sensitivity

    def _data_quality_adjustment(
        self,
        quality_score: float,
        base_se: float,
    ) -> float:
        """Calculate adjustment factor based on data quality."""
        # Lower quality = wider intervals
        # quality_score of 1.0 = no adjustment
        # quality_score of 0.5 = 50% wider intervals

        if quality_score >= 1.0:
            return 0.0
        elif quality_score >= 0.8:
            return 0.1
        elif quality_score >= 0.6:
            return 0.25
        elif quality_score >= 0.4:
            return 0.5
        else:
            return 1.0

    def _generate_interpretation(
        self,
        point: float,
        ci_95: Tuple[float, float],
        method_variation: Dict[str, float],
        sensitivity: Dict[str, Tuple[float, float]],
    ) -> str:
        """Generate human-readable interpretation."""
        width = ci_95[1] - ci_95[0]
        relative_width = width / abs(point) if point != 0 else float('inf')

        # Method agreement
        method_values = list(method_variation.values())
        method_spread = max(method_values) - min(method_values)

        if relative_width < 0.1:
            precision = "high precision"
        elif relative_width < 0.25:
            precision = "moderate precision"
        else:
            precision = "low precision"

        if method_spread < abs(point) * 0.05:
            agreement = "strong agreement across methods"
        elif method_spread < abs(point) * 0.15:
            agreement = "reasonable agreement across methods"
        else:
            agreement = "significant variation across methods"

        interpretation = (
            f"The point estimate of {point:.2f} has {precision} "
            f"(95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]). "
            f"There is {agreement}."
        )

        # Add sensitivity warnings
        max_sensitivity = 0
        most_sensitive = None
        for param, (low, high) in sensitivity.items():
            param_sensitivity = (high - low) / abs(point) if point != 0 else 0
            if param_sensitivity > max_sensitivity:
                max_sensitivity = param_sensitivity
                most_sensitive = param

        if most_sensitive and max_sensitivity > 0.1:
            interpretation += f" The estimate is most sensitive to {most_sensitive}."

        return interpretation

    def monte_carlo_simulation(
        self,
        model: Callable[[Dict[str, float]], float],
        parameter_distributions: Dict[str, Tuple[str, Dict[str, float]]],
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with parameter uncertainty.

        Args:
            model: Function that takes parameters and returns output
            parameter_distributions: Dict of param name to (dist_type, dist_params)
                e.g., {"growth_rate": ("normal", {"loc": 7.0, "scale": 1.5})}

        Returns:
            Dictionary with simulation results
        """
        results = []

        for _ in range(self.monte_carlo_simulations):
            # Sample parameters
            params = {}
            for param_name, (dist_type, dist_params) in parameter_distributions.items():
                if dist_type == "normal":
                    params[param_name] = np.random.normal(**dist_params)
                elif dist_type == "uniform":
                    params[param_name] = np.random.uniform(**dist_params)
                elif dist_type == "triangular":
                    params[param_name] = np.random.triangular(**dist_params)
                elif dist_type == "lognormal":
                    params[param_name] = np.random.lognormal(**dist_params)
                else:
                    raise ValueError(f"Unknown distribution: {dist_type}")

            # Run model
            try:
                result = model(params)
                results.append(result)
            except Exception:
                continue

        results = np.array(results)

        return {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "median": float(np.median(results)),
            "ci_95": (
                float(np.percentile(results, 2.5)),
                float(np.percentile(results, 97.5)),
            ),
            "ci_90": (
                float(np.percentile(results, 5)),
                float(np.percentile(results, 95)),
            ),
            "histogram": np.histogram(results, bins=50),
            "n_simulations": len(results),
        }

    def sensitivity_analysis(
        self,
        model: Callable[[Dict[str, float]], float],
        base_params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 20,
    ) -> Dict[str, SensitivityResult]:
        """
        Perform sensitivity analysis on model parameters.

        Args:
            model: Function that takes parameters and returns output
            base_params: Baseline parameter values
            param_ranges: Dict of param name to (min, max) range
            n_points: Number of points to test per parameter

        Returns:
            Dictionary of parameter name to SensitivityResult
        """
        baseline_result = model(base_params)
        results = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            test_values = np.linspace(min_val, max_val, n_points)
            test_results = []

            for val in test_values:
                test_params = base_params.copy()
                test_params[param_name] = val
                try:
                    test_results.append(model(test_params))
                except Exception:
                    test_results.append(np.nan)

            # Calculate sensitivity coefficient
            valid_mask = ~np.isnan(test_results)
            if valid_mask.sum() > 2:
                slope, _, r_value, _, _ = stats.linregress(
                    test_values[valid_mask],
                    np.array(test_results)[valid_mask]
                )
                sensitivity = slope * (base_params[param_name] / baseline_result)
            else:
                sensitivity = 0.0

            is_sensitive = abs(sensitivity) > 0.1

            if is_sensitive:
                recommendation = f"Results are sensitive to {param_name}. Consider reporting ranges."
            else:
                recommendation = f"Results are robust to {param_name} within tested range."

            results[param_name] = SensitivityResult(
                parameter=param_name,
                baseline_value=base_params[param_name],
                tested_range=test_values.tolist(),
                results=test_results,
                sensitivity_coefficient=sensitivity,
                is_sensitive=is_sensitive,
                recommendation=recommendation,
            )

        return results

    def bayesian_estimate(
        self,
        data: pd.Series,
        prior_mean: float,
        prior_std: float,
    ) -> Dict[str, Any]:
        """
        Bayesian estimation with conjugate prior.

        Args:
            data: Observed data
            prior_mean: Prior mean
            prior_std: Prior standard deviation

        Returns:
            Dictionary with posterior estimates
        """
        # Assume normal-normal conjugate model
        n = len(data)
        data_mean = data.mean()
        data_var = data.var()

        prior_var = prior_std ** 2
        likelihood_var = data_var / n

        # Posterior parameters
        posterior_var = 1 / (1/prior_var + 1/likelihood_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + data_mean/likelihood_var)
        posterior_std = np.sqrt(posterior_var)

        # Credible intervals
        ci_95 = (
            posterior_mean - 1.96 * posterior_std,
            posterior_mean + 1.96 * posterior_std,
        )
        ci_90 = (
            posterior_mean - 1.645 * posterior_std,
            posterior_mean + 1.645 * posterior_std,
        )

        # Prior weight vs data weight
        prior_weight = likelihood_var / (prior_var + likelihood_var)
        data_weight = prior_var / (prior_var + likelihood_var)

        return {
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_95": ci_95,
            "credible_interval_90": ci_90,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "data_mean": data_mean,
            "prior_weight": prior_weight,
            "data_weight": data_weight,
            "shrinkage_toward_prior": prior_weight,
        }

    def combine_estimates(
        self,
        estimates: List[UncertaintyEstimate],
        weights: Optional[List[float]] = None,
    ) -> UncertaintyEstimate:
        """
        Combine multiple estimates using meta-analysis techniques.

        Args:
            estimates: List of UncertaintyEstimate objects
            weights: Optional weights (default: inverse variance weighting)

        Returns:
            Combined UncertaintyEstimate
        """
        if not estimates:
            raise ValueError("No estimates to combine")

        points = np.array([e.point_estimate for e in estimates])
        variances = np.array([e.standard_error ** 2 for e in estimates])

        if weights is None:
            # Inverse variance weighting
            weights = 1 / variances
            weights = weights / weights.sum()

        combined_point = np.average(points, weights=weights)
        combined_var = 1 / np.sum(1 / variances)
        combined_se = np.sqrt(combined_var)

        # Check heterogeneity (I² statistic)
        Q = np.sum((points - combined_point) ** 2 / variances)
        df = len(estimates) - 1
        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

        ci_95 = (combined_point - 1.96 * combined_se, combined_point + 1.96 * combined_se)
        ci_90 = (combined_point - 1.645 * combined_se, combined_point + 1.645 * combined_se)
        ci_80 = (combined_point - 1.28 * combined_se, combined_point + 1.28 * combined_se)

        interpretation = (
            f"Combined estimate from {len(estimates)} sources. "
            f"Heterogeneity (I²) = {I2:.1f}%. "
        )
        if I2 > 50:
            interpretation += "High heterogeneity suggests caution in interpretation."
        elif I2 > 25:
            interpretation += "Moderate heterogeneity present."
        else:
            interpretation += "Low heterogeneity - estimates are consistent."

        return UncertaintyEstimate(
            point_estimate=combined_point,
            standard_error=combined_se,
            confidence_interval_95=ci_95,
            confidence_interval_90=ci_90,
            confidence_interval_80=ci_80,
            distribution="meta-analysis",
            method_variation={"I2_heterogeneity": I2},
            assumption_sensitivity={},
            data_quality_adjustment=0,
            interpretation=interpretation,
        )
