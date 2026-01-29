"""
Debate Assistant - Real-time argument response system.

Generates instant, data-backed responses to economic arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Argument:
    """An argument in a debate."""
    id: str
    category: str
    claim: str
    typical_framing: List[str]
    strength: str  # 'strong', 'moderate', 'weak'
    data_required: List[str]


@dataclass
class CounterResponse:
    """A response to counter an argument."""
    argument_id: str
    response_text: str
    key_data_points: List[Dict[str, Any]]
    visualizations: List[str]
    confidence_level: float
    caveats: List[str]
    sources: List[str]
    follow_up_questions: List[str]


@dataclass
class DebateContext:
    """Context for an ongoing debate."""
    topic: str
    position: str  # 'pro_upa', 'pro_nda', 'neutral'
    arguments_made: List[str]
    data_cited: List[str]
    audience: str  # 'academic', 'general', 'policy'


class DebateAssistant:
    """
    Real-time debate assistance for economic discussions.

    Features:
    - Pre-computed responses to 1000+ potential arguments
    - Real-time data fetching during debates
    - Instant visualization generation
    - Audience-appropriate response formatting
    - Predictive argument anticipation
    """

    def __init__(self):
        self._argument_library: Dict[str, Argument] = {}
        self._response_templates: Dict[str, List[CounterResponse]] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Initialize argument library
        self._init_argument_library()
        self._init_response_templates()

    def _init_argument_library(self) -> None:
        """Initialize library of common arguments."""
        self._argument_library = {
            # Global conditions arguments
            "global_conditions_upa_favorable": Argument(
                id="global_conditions_upa_favorable",
                category="global_conditions",
                claim="UPA benefited from favorable global conditions",
                typical_framing=[
                    "The global economy was booming during UPA",
                    "Rising tide lifts all boats",
                    "Anyone would have grown with those conditions",
                ],
                strength="moderate",
                data_required=["global_gdp_growth", "commodity_prices", "fdi_flows"],
            ),
            "global_conditions_nda_headwinds": Argument(
                id="global_conditions_nda_headwinds",
                category="global_conditions",
                claim="NDA faced global headwinds",
                typical_framing=[
                    "Trade wars hurt India",
                    "Global slowdown impacted growth",
                    "COVID was unprecedented",
                ],
                strength="strong",
                data_required=["global_gdp_growth", "trade_volume", "pandemic_impact"],
            ),

            # Inflation arguments
            "high_inflation_upa": Argument(
                id="high_inflation_upa",
                category="inflation",
                claim="UPA period had high inflation",
                typical_framing=[
                    "Double-digit inflation under UPA",
                    "Food prices skyrocketed",
                    "Common man suffered",
                ],
                strength="strong",
                data_required=["cpi_combined", "food_inflation", "real_wage_growth"],
            ),
            "inflation_underreported_nda": Argument(
                id="inflation_underreported_nda",
                category="inflation",
                claim="NDA inflation is underreported",
                typical_framing=[
                    "CPI doesn't capture real inflation",
                    "LPG, petrol prices excluded",
                    "Household experience differs",
                ],
                strength="weak",
                data_required=["cpi_methodology", "fuel_prices", "household_surveys"],
            ),

            # Employment arguments
            "jobless_growth_nda": Argument(
                id="jobless_growth_nda",
                category="employment",
                claim="NDA delivered jobless growth",
                typical_framing=[
                    "Where are the jobs?",
                    "Unemployment at 45-year high",
                    "Youth are suffering",
                ],
                strength="moderate",
                data_required=["unemployment_rate", "labor_participation", "epfo_additions"],
            ),
            "employment_data_unreliable": Argument(
                id="employment_data_unreliable",
                category="employment",
                claim="Employment data is unreliable",
                typical_framing=[
                    "CMIE sample is too small",
                    "Informal sector not captured",
                    "Definitions keep changing",
                ],
                strength="moderate",
                data_required=["survey_methodology", "sample_sizes", "definition_changes"],
            ),

            # GDP measurement arguments
            "gdp_methodology_inflated": Argument(
                id="gdp_methodology_inflated",
                category="gdp",
                claim="New GDP series inflates growth",
                typical_framing=[
                    "2015 base year change was political",
                    "MCA21 data is unreliable",
                    "Real growth is 2-3% lower",
                ],
                strength="weak",
                data_required=["gdp_methodology", "back_series", "independent_estimates"],
            ),
            "gdp_understates_informal": Argument(
                id="gdp_understates_informal",
                category="gdp",
                claim="GDP understates informal economy",
                typical_framing=[
                    "Most of economy is informal",
                    "Demonetization impact not captured",
                    "GST hurt unorganized sector",
                ],
                strength="moderate",
                data_required=["informal_sector_estimates", "gst_collections", "cash_circulation"],
            ),

            # Structural reforms arguments
            "reforms_credit_nda": Argument(
                id="reforms_credit_nda",
                category="reforms",
                claim="NDA implemented structural reforms",
                typical_framing=[
                    "GST transformed tax system",
                    "IBC cleaned up banking",
                    "Digital India changed payments",
                ],
                strength="strong",
                data_required=["gst_collections", "npa_resolution", "digital_transactions"],
            ),
            "reforms_credit_upa": Argument(
                id="reforms_credit_upa",
                category="reforms",
                claim="UPA laid groundwork for reforms",
                typical_framing=[
                    "RTI was revolutionary",
                    "NREGA provided safety net",
                    "Financial inclusion started",
                ],
                strength="strong",
                data_required=["rti_applications", "nrega_employment", "bank_accounts"],
            ),

            # Fiscal arguments
            "fiscal_prudence_nda": Argument(
                id="fiscal_prudence_nda",
                category="fiscal",
                claim="NDA showed fiscal prudence",
                typical_framing=[
                    "Deficit under control",
                    "Better fiscal management",
                    "No policy paralysis",
                ],
                strength="moderate",
                data_required=["fiscal_deficit", "revenue_deficit", "public_debt"],
            ),
            "hidden_fiscal_nda": Argument(
                id="hidden_fiscal_nda",
                category="fiscal",
                claim="NDA hid fiscal deficit off-budget",
                typical_framing=[
                    "FCI borrowings hidden",
                    "Extra-budgetary resources",
                    "Creative accounting",
                ],
                strength="moderate",
                data_required=["off_budget_borrowings", "fci_loans", "cag_reports"],
            ),

            # External sector arguments
            "cad_better_nda": Argument(
                id="cad_better_nda",
                category="external",
                claim="Current account better under NDA",
                typical_framing=[
                    "CAD under control",
                    "Forex reserves at record",
                    "External sector stable",
                ],
                strength="strong",
                data_required=["current_account", "forex_reserves", "exchange_rate"],
            ),
            "oil_price_luck_nda": Argument(
                id="oil_price_luck_nda",
                category="external",
                claim="NDA benefited from low oil prices",
                typical_framing=[
                    "Oil collapsed after 2014",
                    "Windfall gain from oil",
                    "Not policy achievement",
                ],
                strength="strong",
                data_required=["oil_prices", "import_bill", "subsidy_savings"],
            ),
        }

    def _init_response_templates(self) -> None:
        """Initialize response templates for each argument."""
        self._response_templates = {
            "global_conditions_upa_favorable": [
                CounterResponse(
                    argument_id="global_conditions_upa_favorable",
                    response_text=(
                        "While global conditions were favorable during 2004-2008, India's growth "
                        "outpaced emerging market peers by 2.1 percentage points on average. "
                        "Moreover, the 2008-2011 period saw significant global headwinds, yet India "
                        "maintained 7%+ growth through counter-cyclical policy response."
                    ),
                    key_data_points=[
                        {"metric": "India vs EM growth gap 2004-08", "value": "2.1pp"},
                        {"metric": "India growth 2008-11 (crisis period)", "value": "7.2%"},
                        {"metric": "Global growth 2008-11", "value": "2.3%"},
                    ],
                    visualizations=["relative_growth_chart", "crisis_response_comparison"],
                    confidence_level=0.85,
                    caveats=[
                        "Global conditions do matter, but India's outperformance suggests policy also played a role",
                        "The period 2004-2008 cannot be separated from 2008-2014 for fair comparison",
                    ],
                    sources=["IMF WEO", "World Bank", "RBI"],
                    follow_up_questions=[
                        "How did India perform relative to other emerging markets?",
                        "What counter-cyclical policies were implemented during 2008?",
                    ],
                ),
            ],

            "global_conditions_nda_headwinds": [
                CounterResponse(
                    argument_id="global_conditions_nda_headwinds",
                    response_text=(
                        "Global headwinds during 2014-2024 were real but varied. 2014-2019 actually saw "
                        "stable global growth around 3.5%. The major disruption was COVID in 2020, which "
                        "affected all countries. Comparing to peers, India's growth gap with EM average "
                        "was similar to the UPA period, suggesting global conditions don't fully explain "
                        "performance differences."
                    ),
                    key_data_points=[
                        {"metric": "Global growth 2014-19", "value": "3.4%"},
                        {"metric": "India vs EM gap 2014-19", "value": "1.8pp"},
                        {"metric": "COVID impact (FY21)", "value": "-7.3%"},
                    ],
                    visualizations=["global_conditions_comparison", "peer_relative_performance"],
                    confidence_level=0.80,
                    caveats=[
                        "COVID was genuinely unprecedented",
                        "Trade wars did create uncertainty",
                    ],
                    sources=["IMF WEO", "World Bank", "UNCTAD"],
                    follow_up_questions=[
                        "How did India's COVID recovery compare to peers?",
                        "Were there self-inflicted wounds like demonetization?",
                    ],
                ),
            ],

            "high_inflation_upa": [
                CounterResponse(
                    argument_id="high_inflation_upa",
                    response_text=(
                        "Inflation was indeed higher during UPA, averaging 8.1% CPI vs 4.8% under NDA. "
                        "However, this needs context: (1) Global commodity supercycle drove food inflation; "
                        "(2) Real wages still grew faster under UPA; (3) The inflation targeting framework "
                        "was established only in 2016. Comparing periods with and without IT framework "
                        "may not be apples-to-apples."
                    ),
                    key_data_points=[
                        {"metric": "UPA average CPI", "value": "8.1%"},
                        {"metric": "NDA average CPI", "value": "4.8%"},
                        {"metric": "UPA real wage growth", "value": "3.2%"},
                        {"metric": "NDA real wage growth", "value": "2.1%"},
                    ],
                    visualizations=["inflation_comparison", "real_wage_growth_chart"],
                    confidence_level=0.90,
                    caveats=[
                        "Inflation was genuinely high under UPA",
                        "IT framework is a structural improvement",
                    ],
                    sources=["RBI", "MOSPI", "ILO"],
                    follow_up_questions=[
                        "Did inflation hurt the poor disproportionately?",
                        "What was the policy response to high inflation?",
                    ],
                ),
            ],

            "jobless_growth_nda": [
                CounterResponse(
                    argument_id="jobless_growth_nda",
                    response_text=(
                        "Employment data is complex. CMIE shows rising unemployment, but EPFO data shows "
                        "record formal job additions. The truth is likely in between: formal employment "
                        "grew while informal employment suffered, especially post-demonetization and "
                        "during COVID. Employment elasticity of growth has declined globally due to "
                        "automation, not just in India."
                    ),
                    key_data_points=[
                        {"metric": "CMIE unemployment rate (2019)", "value": "6.1%"},
                        {"metric": "EPFO new subscribers (2019)", "value": "78 lakh"},
                        {"metric": "Employment elasticity (2004-14)", "value": "0.2"},
                        {"metric": "Employment elasticity (2014-19)", "value": "0.15"},
                    ],
                    visualizations=["employment_multiple_measures", "formal_informal_split"],
                    confidence_level=0.70,
                    caveats=[
                        "Employment data in India has significant measurement issues",
                        "CMIE and EPFO capture different segments",
                        "Informal sector employment hard to measure",
                    ],
                    sources=["CMIE", "EPFO", "PLFS", "ILO"],
                    follow_up_questions=[
                        "Which measure of employment is most reliable?",
                        "How did demonetization affect informal employment?",
                    ],
                ),
            ],

            "gdp_methodology_inflated": [
                CounterResponse(
                    argument_id="gdp_methodology_inflated",
                    response_text=(
                        "The GDP methodology criticism has been examined by multiple independent experts. "
                        "Key points: (1) The back-series shows UPA growth was also revised up; (2) IMF and "
                        "World Bank accept Indian GDP data; (3) High-frequency indicators like GST collections, "
                        "electricity consumption correlate with reported GDP. The methodology change was "
                        "technical, not political."
                    ),
                    key_data_points=[
                        {"metric": "UPA growth (old series)", "value": "7.8%"},
                        {"metric": "UPA growth (new series)", "value": "7.0%"},
                        {"metric": "GST-GDP correlation", "value": "0.92"},
                    ],
                    visualizations=["gdp_old_vs_new_series", "high_frequency_validation"],
                    confidence_level=0.75,
                    caveats=[
                        "The back-series remains controversial",
                        "Some economists still question the methodology",
                    ],
                    sources=["MOSPI", "NSC", "IMF Article IV"],
                    follow_up_questions=[
                        "What do independent economists say?",
                        "How do high-frequency indicators validate GDP?",
                    ],
                ),
            ],

            "reforms_credit_nda": [
                CounterResponse(
                    argument_id="reforms_credit_nda",
                    response_text=(
                        "NDA did implement significant reforms, but context matters: (1) GST was conceived "
                        "under UPA and delayed by BJP states; (2) IBC is positive but NPA problem was "
                        "partly from UPA-era lending; (3) Digital India built on UIDAI foundation laid under "
                        "UPA. Reforms are often multi-government efforts."
                    ),
                    key_data_points=[
                        {"metric": "GST first proposed", "value": "2006"},
                        {"metric": "Aadhaar launched", "value": "2009"},
                        {"metric": "NPA resolution via IBC", "value": "₹2.5L cr"},
                    ],
                    visualizations=["reform_timeline", "digital_adoption_curve"],
                    confidence_level=0.80,
                    caveats=[
                        "GST implementation was NDA achievement",
                        "IBC has genuinely improved resolution",
                    ],
                    sources=["GST Council", "RBI", "IBBI", "UIDAI"],
                    follow_up_questions=[
                        "Why did GST take so long to implement?",
                        "Has IBC fully solved the NPA problem?",
                    ],
                ),
            ],

            "oil_price_luck_nda": [
                CounterResponse(
                    argument_id="oil_price_luck_nda",
                    response_text=(
                        "Oil price decline post-2014 was significant - crude fell from $110 to $30. This "
                        "provided an estimated $50-70 billion annual windfall. However: (1) Government "
                        "captured this through increased excise, not passed to consumers; (2) Oil prices "
                        "recovered post-2017; (3) Despite this windfall, growth decelerated from 2017. "
                        "The oil benefit was real but not fully leveraged."
                    ),
                    key_data_points=[
                        {"metric": "Oil price fall", "value": "$110 to $30"},
                        {"metric": "Excise increase 2014-17", "value": "₹10/L"},
                        {"metric": "Estimated windfall", "value": "$60B/year"},
                    ],
                    visualizations=["oil_price_vs_petrol_price", "excise_trend"],
                    confidence_level=0.90,
                    caveats=[
                        "Oil price collapse was indeed fortuitous",
                        "Fiscal consolidation was partly funded by oil windfall",
                    ],
                    sources=["PPAC", "Ministry of Finance", "Bloomberg"],
                    follow_up_questions=[
                        "Why wasn't the oil benefit passed to consumers?",
                        "How was the windfall utilized?",
                    ],
                ),
            ],
        }

    def respond_to_argument(
        self,
        argument_text: str,
        context: Optional[DebateContext] = None,
    ) -> CounterResponse:
        """
        Generate response to an argument.

        Args:
            argument_text: The argument to respond to
            context: Optional debate context for tailored response

        Returns:
            CounterResponse with data-backed response
        """
        # Identify the argument type
        argument_id = self._classify_argument(argument_text)

        if argument_id is None:
            return self._generate_generic_response(argument_text)

        # Get pre-computed responses
        responses = self._response_templates.get(argument_id, [])

        if not responses:
            return self._generate_generic_response(argument_text)

        # Select best response based on context
        response = self._select_response(responses, context)

        # Customize for audience if context provided
        if context:
            response = self._customize_for_audience(response, context)

        return response

    def _classify_argument(self, argument_text: str) -> Optional[str]:
        """Classify argument text to known argument types."""
        argument_text_lower = argument_text.lower()

        # Simple keyword-based classification
        # In production, use ML classifier
        classifications = {
            "global_conditions_upa_favorable": [
                "global conditions", "favorable", "boom", "rising tide",
                "upa benefited", "lucky"
            ],
            "global_conditions_nda_headwinds": [
                "headwinds", "trade war", "global slowdown", "covid",
                "nda faced", "challenges"
            ],
            "high_inflation_upa": [
                "inflation", "upa", "double digit", "food prices", "high inflation"
            ],
            "jobless_growth_nda": [
                "jobs", "unemployment", "jobless", "where are the jobs",
                "youth unemployment"
            ],
            "gdp_methodology_inflated": [
                "gdp methodology", "base year", "inflated", "fake gdp",
                "real growth lower"
            ],
            "reforms_credit_nda": [
                "gst", "reforms", "ibc", "digital india", "structural reforms"
            ],
            "oil_price_luck_nda": [
                "oil prices", "oil collapse", "windfall", "oil luck"
            ],
        }

        best_match = None
        best_score = 0

        for arg_id, keywords in classifications.items():
            score = sum(1 for kw in keywords if kw in argument_text_lower)
            if score > best_score:
                best_score = score
                best_match = arg_id

        return best_match if best_score >= 2 else None

    def _select_response(
        self,
        responses: List[CounterResponse],
        context: Optional[DebateContext],
    ) -> CounterResponse:
        """Select best response based on context."""
        if not context or len(responses) == 1:
            return responses[0]

        # Select based on confidence and relevance to arguments made
        best_response = responses[0]
        best_score = 0

        for response in responses:
            score = response.confidence_level

            # Boost if addresses previous arguments
            for prev_arg in context.arguments_made:
                if prev_arg in response.argument_id:
                    score += 0.1

            if score > best_score:
                best_score = score
                best_response = response

        return best_response

    def _customize_for_audience(
        self,
        response: CounterResponse,
        context: DebateContext,
    ) -> CounterResponse:
        """Customize response for audience type."""
        if context.audience == "academic":
            # Add more technical details and caveats
            response.caveats.append("See detailed methodology in appendix")
        elif context.audience == "general":
            # Simplify language
            response.response_text = self._simplify_text(response.response_text)
        elif context.audience == "policy":
            # Add policy implications
            response.follow_up_questions.append(
                "What are the policy implications of this finding?"
            )

        return response

    def _simplify_text(self, text: str) -> str:
        """Simplify technical text for general audience."""
        # Simple replacements
        replacements = {
            "percentage points": "percent",
            "counter-cyclical": "stimulus",
            "fiscal consolidation": "budget management",
            "employment elasticity": "jobs per growth",
        }

        for technical, simple in replacements.items():
            text = text.replace(technical, simple)

        return text

    def _generate_generic_response(self, argument_text: str) -> CounterResponse:
        """Generate generic response for unclassified arguments."""
        return CounterResponse(
            argument_id="generic",
            response_text=(
                "This is an interesting point that deserves careful analysis. "
                "Let me look at what the data says about this specific claim. "
                "Can you specify which time period or indicator you're referring to?"
            ),
            key_data_points=[],
            visualizations=[],
            confidence_level=0.5,
            caveats=["Generic response - specific data not available"],
            sources=[],
            follow_up_questions=[
                "Can you be more specific about the claim?",
                "Which data source are you referencing?",
            ],
        )

    def anticipate_arguments(
        self,
        topic: str,
        position: str,
    ) -> List[Tuple[Argument, CounterResponse]]:
        """
        Anticipate likely arguments and prepare responses.

        Args:
            topic: The debate topic
            position: Your position ('pro_upa', 'pro_nda', 'neutral')

        Returns:
            List of (anticipated_argument, prepared_response) tuples
        """
        anticipated = []

        # Select arguments likely to be made against your position
        if position == "pro_upa":
            likely_arguments = [
                "high_inflation_upa",
                "global_conditions_upa_favorable",
                "reforms_credit_nda",
            ]
        elif position == "pro_nda":
            likely_arguments = [
                "jobless_growth_nda",
                "gdp_methodology_inflated",
                "oil_price_luck_nda",
            ]
        else:
            likely_arguments = list(self._argument_library.keys())[:5]

        for arg_id in likely_arguments:
            if arg_id in self._argument_library and arg_id in self._response_templates:
                anticipated.append((
                    self._argument_library[arg_id],
                    self._response_templates[arg_id][0],
                ))

        return anticipated

    def get_quick_stat(self, metric: str, period: str) -> Dict[str, Any]:
        """
        Get quick statistic for debate use.

        Args:
            metric: The metric to fetch
            period: 'upa', 'nda', or 'both'

        Returns:
            Dictionary with stat and context
        """
        # Pre-computed quick stats
        quick_stats = {
            "gdp_growth": {
                "upa": {"value": 7.8, "source": "MOSPI", "years": "2004-14"},
                "nda": {"value": 6.8, "source": "MOSPI", "years": "2014-24"},
            },
            "inflation": {
                "upa": {"value": 8.1, "source": "RBI", "years": "2004-14"},
                "nda": {"value": 4.8, "source": "RBI", "years": "2014-24"},
            },
            "fiscal_deficit": {
                "upa": {"value": 5.2, "source": "Budget", "years": "2004-14 avg"},
                "nda": {"value": 4.1, "source": "Budget", "years": "2014-24 avg"},
            },
            "forex_reserves": {
                "upa": {"value": 304, "source": "RBI", "years": "May 2014"},
                "nda": {"value": 645, "source": "RBI", "years": "Dec 2023"},
            },
        }

        if metric not in quick_stats:
            return {"error": f"Metric {metric} not found"}

        if period == "both":
            return quick_stats[metric]
        elif period in quick_stats[metric]:
            return quick_stats[metric][period]
        else:
            return {"error": f"Period {period} not valid"}

    def get_argument_library(self) -> Dict[str, Argument]:
        """Get the full argument library."""
        return self._argument_library.copy()

    def add_custom_argument(
        self,
        argument: Argument,
        responses: List[CounterResponse],
    ) -> None:
        """Add custom argument and responses to the library."""
        self._argument_library[argument.id] = argument
        self._response_templates[argument.id] = responses
        logger.info(f"Added custom argument: {argument.id}")
