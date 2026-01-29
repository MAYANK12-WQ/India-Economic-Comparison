"""
Command Line Interface for India Economic Comparison.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="India Economic Comparison System - Compare economic performance 2004-2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison
  india-econ compare --output report.json

  # Compare specific indicators
  india-econ compare --indicators gdp_growth_rate inflation --output report.json

  # Analyze a specific policy
  india-econ causal --policy demonetization

  # Run counterfactual scenario
  india-econ counterfactual --scenario no_demonetization

  # Start API server
  india-econ serve --port 8000

  # Start interactive dashboard
  india-econ dashboard --port 8050

  # Validate political claims
  india-econ validate-claims --party BJP

  # Get debate response
  india-econ debate "GDP growth was higher under UPA"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Run full comparison")
    compare_parser.add_argument(
        "--indicators", nargs="+", help="Specific indicators to compare"
    )
    compare_parser.add_argument(
        "--output", "-o", type=Path, default=Path("comparison_report.json"),
        help="Output file path"
    )
    compare_parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Output format"
    )
    compare_parser.add_argument(
        "--no-causal", action="store_true", help="Skip causal analysis"
    )
    compare_parser.add_argument(
        "--no-counterfactual", action="store_true", help="Skip counterfactual scenarios"
    )

    # Causal command
    causal_parser = subparsers.add_parser("causal", help="Analyze policy impact")
    causal_parser.add_argument(
        "--policy", required=True,
        choices=["demonetization", "gst_implementation", "covid_lockdown", "2008_crisis"],
        help="Policy to analyze"
    )
    causal_parser.add_argument(
        "--method", default="did",
        choices=["did", "scm", "rdd"],
        help="Causal inference method"
    )

    # Counterfactual command
    cf_parser = subparsers.add_parser("counterfactual", help="Run counterfactual scenario")
    cf_parser.add_argument(
        "--scenario", required=True,
        choices=["no_demonetization", "2008_in_2014", "early_gst", "covid_in_2009",
                 "equal_global_conditions", "equal_demographics"],
        help="Scenario to simulate"
    )
    cf_parser.add_argument(
        "--monte-carlo", type=int, default=1000,
        help="Number of Monte Carlo iterations"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start interactive dashboard")
    dash_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    dash_parser.add_argument("--port", type=int, default=8050, help="Port to bind")

    # Validate claims command
    validate_parser = subparsers.add_parser("validate-claims", help="Validate political claims")
    validate_parser.add_argument("--party", help="Filter by party")
    validate_parser.add_argument("--category", help="Filter by category")
    validate_parser.add_argument("--claim-id", help="Specific claim ID")

    # Debate command
    debate_parser = subparsers.add_parser("debate", help="Get debate response")
    debate_parser.add_argument("argument", help="Argument to respond to")
    debate_parser.add_argument("--audience", default="general",
                               choices=["general", "academic", "policy"])

    # Quick stat command
    stat_parser = subparsers.add_parser("quick-stat", help="Get quick statistic")
    stat_parser.add_argument("metric", help="Metric name")
    stat_parser.add_argument("--period", default="both", choices=["upa", "nda", "both"])

    # Data quality command
    quality_parser = subparsers.add_parser("data-quality", help="Assess data quality")
    quality_parser.add_argument("--indicator", help="Specific indicator")
    quality_parser.add_argument("--source", help="Specific source")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "compare":
            run_comparison(args)
        elif args.command == "causal":
            run_causal_analysis(args)
        elif args.command == "counterfactual":
            run_counterfactual(args)
        elif args.command == "serve":
            run_server(args)
        elif args.command == "dashboard":
            run_dashboard(args)
        elif args.command == "validate-claims":
            run_validate_claims(args)
        elif args.command == "debate":
            run_debate(args)
        elif args.command == "quick-stat":
            run_quick_stat(args)
        elif args.command == "data-quality":
            run_data_quality(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def run_comparison(args):
    """Run full comparison."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()

    logger.info("Running comprehensive comparison...")
    result = engine.run_full_comparison(
        indicators=args.indicators,
        include_causal=not args.no_causal,
        include_counterfactual=not args.no_counterfactual,
    )

    engine.export_report(args.output, args.format)

    print("\n" + "=" * 60)
    print(result.executive_summary)
    print("=" * 60)
    print(f"\nFull report saved to: {args.output}")


def run_causal_analysis(args):
    """Run causal analysis for a policy."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()

    logger.info(f"Analyzing causal impact of {args.policy}...")

    # Fetch required data
    data = engine._fetch_all_data(
        ["gdp_growth_rate", "unemployment_rate", "inflation"],
        engine.UPA_PERIOD if args.policy == "2008_crisis" else engine.NDA_PERIOD,
        engine.NDA_PERIOD,
    )

    effect = engine.causal_engine.analyze_policy_impact(
        args.policy, data, method=args.method
    )

    print("\n" + "=" * 60)
    print(f"CAUSAL ANALYSIS: {args.policy.upper()}")
    print("=" * 60)
    print(f"Method: {effect.method}")
    print(f"Effect: {effect.point_estimate:.3f}")
    print(f"95% CI: [{effect.confidence_interval[0]:.3f}, {effect.confidence_interval[1]:.3f}]")
    print(f"P-value: {effect.p_value:.4f}")
    print(f"Significant: {'Yes' if effect.statistically_significant else 'No'}")
    print(f"\nInterpretation:\n{effect.interpretation}")
    print("=" * 60)


def run_counterfactual(args):
    """Run counterfactual scenario."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()

    logger.info(f"Running counterfactual scenario: {args.scenario}...")

    data = engine._fetch_all_data(
        ["gdp_growth_rate"],
        engine.UPA_PERIOD,
        engine.NDA_PERIOD,
    )

    result = engine.counterfactual_sim.simulate(
        args.scenario, data, monte_carlo_runs=args.monte_carlo
    )

    print("\n" + "=" * 60)
    print(f"COUNTERFACTUAL SCENARIO: {args.scenario.upper()}")
    print("=" * 60)
    print(f"Description: {result.scenario.description}")
    print(f"\nCumulative Impact: {result.cumulative_impact:.2f}")
    print(f"Annual Impact: {result.annual_impact:.2f}")
    print(f"95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
    print(f"\nKey Insights:")
    for insight in result.key_insights:
        print(f"  - {insight}")
    print("=" * 60)


def run_server(args):
    """Start API server."""
    try:
        import uvicorn
        from .api.rest_api import app

        logger.info(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


def run_dashboard(args):
    """Start interactive dashboard."""
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    print("Dashboard feature requires additional setup.")
    print("Visit http://localhost:8050 after starting.")
    # In production, this would start a Dash app


def run_validate_claims(args):
    """Validate political claims."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()

    data = engine._fetch_all_data(
        ["gdp_growth_rate", "farmer_income"],
        engine.UPA_PERIOD,
        engine.NDA_PERIOD,
    )

    validations = engine.predictive_forensics.validate_all_claims(
        data, party=args.party, category=args.category
    )
    scorecard = engine.predictive_forensics.get_claim_scorecard(validations)

    print("\n" + "=" * 60)
    print("CLAIM VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total claims analyzed: {scorecard['total_claims']}")
    print(f"Met: {scorecard['status_breakdown']['met']}")
    print(f"Partially Met: {scorecard['status_breakdown']['partially_met']}")
    print(f"Not Met: {scorecard['status_breakdown']['not_met']}")
    print(f"Cannot Verify: {scorecard['status_breakdown']['cannot_verify']}")
    print(f"\nMet Rate: {scorecard['met_rate_pct']:.1f}%")
    print("=" * 60)


def run_debate(args):
    """Get debate response."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()
    response = engine.get_debate_response(args.argument)

    print("\n" + "=" * 60)
    print("DEBATE RESPONSE")
    print("=" * 60)
    print(f"Argument: {args.argument}")
    print(f"\nResponse:\n{response.response_text}")
    print(f"\nKey Data Points:")
    for dp in response.key_data_points:
        print(f"  - {dp['metric']}: {dp['value']}")
    print(f"\nConfidence: {response.confidence_level:.0%}")
    print(f"\nCaveats:")
    for caveat in response.caveats:
        print(f"  - {caveat}")
    print("=" * 60)


def run_quick_stat(args):
    """Get quick statistic."""
    from .debate_assistant import DebateAssistant

    assistant = DebateAssistant()
    stat = assistant.get_quick_stat(args.metric, args.period)

    print("\n" + "=" * 60)
    print(f"QUICK STAT: {args.metric.upper()}")
    print("=" * 60)
    if "error" in stat:
        print(f"Error: {stat['error']}")
    elif args.period == "both":
        print(f"UPA (2004-14): {stat.get('upa', {}).get('value', 'N/A')}")
        print(f"NDA (2014-24): {stat.get('nda', {}).get('value', 'N/A')}")
    else:
        print(f"{args.period.upper()}: {stat.get('value', 'N/A')}")
    print("=" * 60)


def run_data_quality(args):
    """Assess data quality."""
    from .comparison_engine import ComparisonEngine

    engine = ComparisonEngine()

    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    print("Source Reliability Scores:")
    print("  - MOSPI: 8.0/10 (Official government statistics)")
    print("  - RBI: 9.0/10 (Central bank data)")
    print("  - CMIE: 7.0/10 (Private survey data)")
    print("  - World Bank: 9.0/10 (International benchmark)")
    print("  - IMF: 9.0/10 (International benchmark)")
    print("\nKnown Methodology Changes:")
    print("  - 2015: GDP base year changed to 2011-12")
    print("  - 2017: Employment survey methodology changed")
    print("=" * 60)


if __name__ == "__main__":
    main()
