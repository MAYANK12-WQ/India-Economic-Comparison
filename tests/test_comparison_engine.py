"""
Tests for the Comparison Engine.
"""

import pytest
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


class TestComparisonEngine:
    """Test suite for ComparisonEngine."""

    def test_period_definition(self):
        """Test that periods are correctly defined."""
        from src.comparison_engine import ComparisonEngine

        engine = ComparisonEngine()

        assert engine.UPA_PERIOD.name == "upa"
        assert engine.UPA_PERIOD.start_date == datetime(2004, 5, 22)
        assert engine.UPA_PERIOD.end_date == datetime(2014, 5, 26)

        assert engine.NDA_PERIOD.name == "nda"
        assert engine.NDA_PERIOD.start_date == datetime(2014, 5, 26)

    def test_mock_data_generation(self):
        """Test mock data generation."""
        from src.comparison_engine import ComparisonEngine

        engine = ComparisonEngine()

        data = engine._generate_mock_data(
            "gdp_growth_rate",
            datetime(2004, 1, 1),
            datetime(2024, 1, 1),
        )

        assert isinstance(data, pd.DataFrame)
        assert "value" in data.columns
        assert len(data) > 0

    def test_interpretation_generation(self):
        """Test interpretation text generation."""
        from src.comparison_engine import ComparisonEngine

        engine = ComparisonEngine()

        interpretation = engine._interpret_comparison(
            "gdp_growth_rate",
            mean_a=7.8,
            mean_b=6.5,
            p_value=0.01,
        )

        assert "gdp_growth_rate" in interpretation
        assert "statistically significant" in interpretation


class TestDataSkepticism:
    """Test suite for DataSkepticismEngine."""

    def test_quality_assessment(self):
        """Test data quality assessment."""
        from src.data_skepticism import DataSkepticismEngine

        engine = DataSkepticismEngine()

        # Create test data
        dates = pd.date_range("2010-01-01", "2020-01-01", freq="Q")
        data = pd.DataFrame({
            "value": np.random.normal(7, 1, len(dates))
        }, index=dates)

        report = engine.assess_data_quality(data, "gdp_growth_rate", "MOSPI")

        assert report.overall_score >= 0
        assert report.overall_score <= 10
        assert report.indicator == "gdp_growth_rate"
        assert report.source == "MOSPI"

    def test_bias_detection(self):
        """Test bias detection."""
        from src.data_skepticism import DataSkepticismEngine

        engine = DataSkepticismEngine()

        dates = pd.date_range("2010-01-01", "2020-01-01", freq="Q")
        data = pd.DataFrame({
            "value": np.random.normal(7, 1, len(dates))
        }, index=dates)

        biases = engine.detect_potential_biases(data, "gdp_growth_rate", "MOSPI")

        assert isinstance(biases, list)


class TestCausalInference:
    """Test suite for CausalInferenceEngine."""

    def test_policy_analysis(self):
        """Test policy impact analysis."""
        from src.causal_inference import CausalInferenceEngine

        engine = CausalInferenceEngine()

        # Create test data
        dates = pd.date_range("2014-01-01", "2020-01-01", freq="Q")
        data = {
            "gdp_growth_rate": pd.DataFrame({
                "value": np.random.normal(7, 1, len(dates))
            }, index=dates)
        }

        effect = engine.analyze_policy_impact(
            "demonetization",
            data,
            method="did",
            pre_periods=4,
            post_periods=4,
        )

        assert effect.treatment == "Demonetization"
        assert effect.method == "Difference-in-Differences"
        assert effect.p_value >= 0
        assert effect.p_value <= 1


class TestCounterfactual:
    """Test suite for CounterfactualSimulator."""

    def test_scenario_listing(self):
        """Test scenario listing."""
        from src.counterfactual import CounterfactualSimulator

        sim = CounterfactualSimulator()
        scenarios = sim.list_scenarios()

        assert "no_demonetization" in scenarios
        assert "equal_global_conditions" in scenarios

    def test_simulation(self):
        """Test counterfactual simulation."""
        from src.counterfactual import CounterfactualSimulator

        sim = CounterfactualSimulator()

        dates = pd.date_range("2014-01-01", "2020-01-01", freq="Q")
        data = {
            "gdp_growth_rate": pd.DataFrame({
                "value": np.random.normal(7, 1, len(dates))
            }, index=dates)
        }

        result = sim.simulate("no_demonetization", data, monte_carlo_runs=100)

        assert result.scenario.name == "No Demonetization"
        assert isinstance(result.cumulative_impact, float)


class TestDebateAssistant:
    """Test suite for DebateAssistant."""

    def test_argument_response(self):
        """Test argument response generation."""
        from src.debate_assistant import DebateAssistant

        assistant = DebateAssistant()

        response = assistant.respond_to_argument(
            "UPA benefited from favorable global conditions"
        )

        assert response is not None
        assert len(response.response_text) > 0

    def test_quick_stat(self):
        """Test quick stat retrieval."""
        from src.debate_assistant import DebateAssistant

        assistant = DebateAssistant()

        stat = assistant.get_quick_stat("gdp_growth", "both")

        assert "upa" in stat or "nda" in stat


class TestEthicalFramework:
    """Test suite for EthicalFramework."""

    def test_ethical_review(self):
        """Test ethical review."""
        from src.ethical_framework import EthicalFramework

        framework = EthicalFramework()

        analysis = {
            "metrics": ["gdp_growth_rate", "inflation", "unemployment"],
            "period_a_analysis": {"metrics": ["a", "b", "c"]},
            "period_b_analysis": {"metrics": ["a", "b", "c"]},
            "assumptions": ["Test assumption"],
            "limitations": ["Test limitation"],
        }

        report = framework.ethical_review(analysis)

        assert report.overall_score >= 0
        assert report.overall_score <= 10


class TestUncertainty:
    """Test suite for UncertaintyQuantifier."""

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification."""
        from src.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier(bootstrap_iterations=100)

        data = pd.DataFrame({
            "value": np.random.normal(7, 1, 50)
        })

        def estimator(df):
            return df["value"].mean()

        result = quantifier.quantify_uncertainty(data, estimator)

        assert result.point_estimate is not None
        assert result.confidence_interval_95[0] < result.confidence_interval_95[1]


class TestPredictiveForensics:
    """Test suite for PredictiveForensics."""

    def test_claim_listing(self):
        """Test claim listing."""
        from src.predictive_forensics import PredictiveForensics

        forensics = PredictiveForensics()

        economic_claims = forensics.get_claims_by_category("economic")

        assert len(economic_claims) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
