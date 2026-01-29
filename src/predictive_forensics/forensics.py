"""
Predictive Forensics - Testing claims against outcomes.

Validates political promises and economic predictions.
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
class PoliticalClaim:
    """A political or economic claim to validate."""
    id: str
    claim_text: str
    claim_date: datetime
    claimed_by: str
    party: str
    target_metric: str
    target_value: Optional[float]
    target_date: Optional[datetime]
    source: str
    category: str  # 'economic', 'social', 'infrastructure', etc.


@dataclass
class ClaimValidation:
    """Validation result for a claim."""
    claim: PoliticalClaim
    status: str  # 'met', 'partially_met', 'not_met', 'ongoing', 'cannot_verify'
    actual_value: Optional[float]
    achievement_pct: Optional[float]
    evidence: List[str]
    data_sources: List[str]
    methodology_notes: str
    confidence_level: float
    caveats: List[str]


class PredictiveForensics:
    """
    Validates political and economic claims against actual outcomes.

    Tracks:
    - Election promises vs delivery
    - Economic projections vs actuals
    - Policy goals vs achievements
    """

    def __init__(self):
        self._claims: Dict[str, PoliticalClaim] = {}
        self._validations: Dict[str, ClaimValidation] = {}
        self._data_store: Dict[str, pd.DataFrame] = {}

        # Initialize known claims
        self._init_known_claims()

    def _init_known_claims(self) -> None:
        """Initialize database of known claims."""
        self._claims = {
            # UPA Era Claims
            "upa_india_shining": PoliticalClaim(
                id="upa_india_shining",
                claim_text="India Shining - 8%+ growth will continue",
                claim_date=datetime(2004, 1, 1),
                claimed_by="NDA Campaign",
                party="BJP",
                target_metric="gdp_growth_rate",
                target_value=8.0,
                target_date=datetime(2009, 1, 1),
                source="2004 Election Campaign",
                category="economic",
            ),
            "upa_inclusive_growth": PoliticalClaim(
                id="upa_inclusive_growth",
                claim_text="Inclusive growth - benefits will reach the poor",
                claim_date=datetime(2004, 5, 22),
                claimed_by="UPA Government",
                party="INC",
                target_metric="poverty_rate",
                target_value=None,  # Reduction target
                target_date=datetime(2014, 1, 1),
                source="Common Minimum Programme",
                category="social",
            ),

            # NDA Era Claims
            "nda_double_farmer_income": PoliticalClaim(
                id="nda_double_farmer_income",
                claim_text="Double farmer incomes by 2022",
                claim_date=datetime(2016, 2, 28),
                claimed_by="PM Modi",
                party="BJP",
                target_metric="farmer_income",
                target_value=200,  # % of baseline
                target_date=datetime(2022, 3, 31),
                source="Budget 2016 Speech",
                category="economic",
            ),
            "nda_5_trillion_economy": PoliticalClaim(
                id="nda_5_trillion_economy",
                claim_text="$5 trillion economy by 2024",
                claim_date=datetime(2019, 7, 5),
                claimed_by="PM Modi",
                party="BJP",
                target_metric="gdp_usd",
                target_value=5.0,  # Trillion USD
                target_date=datetime(2024, 3, 31),
                source="Budget 2019 Speech",
                category="economic",
            ),
            "nda_2_crore_jobs": PoliticalClaim(
                id="nda_2_crore_jobs",
                claim_text="Create 2 crore jobs per year",
                claim_date=datetime(2014, 4, 1),
                claimed_by="BJP Manifesto",
                party="BJP",
                target_metric="jobs_created",
                target_value=20000000,
                target_date=datetime(2019, 5, 1),
                source="2014 Manifesto",
                category="employment",
            ),
            "nda_black_money_return": PoliticalClaim(
                id="nda_black_money_return",
                claim_text="Bring back black money, ₹15 lakh per account",
                claim_date=datetime(2014, 4, 1),
                claimed_by="Campaign",
                party="BJP",
                target_metric="black_money_recovered",
                target_value=1500000,
                target_date=datetime(2019, 5, 1),
                source="Campaign Speech",
                category="fiscal",
            ),
            "nda_acche_din": PoliticalClaim(
                id="nda_acche_din",
                claim_text="Acche Din - Good days are coming",
                claim_date=datetime(2014, 4, 1),
                claimed_by="Campaign",
                party="BJP",
                target_metric="quality_of_life_index",
                target_value=None,
                target_date=datetime(2019, 5, 1),
                source="Campaign Slogan",
                category="social",
            ),

            # Economic Projections
            "imf_2019_projection": PoliticalClaim(
                id="imf_2019_projection",
                claim_text="India GDP growth 7.5% in 2019-20",
                claim_date=datetime(2019, 4, 1),
                claimed_by="IMF",
                party="International",
                target_metric="gdp_growth_rate",
                target_value=7.5,
                target_date=datetime(2020, 3, 31),
                source="IMF WEO April 2019",
                category="economic",
            ),
            "rbi_inflation_target": PoliticalClaim(
                id="rbi_inflation_target",
                claim_text="Inflation target 4% ± 2%",
                claim_date=datetime(2016, 8, 5),
                claimed_by="RBI",
                party="RBI",
                target_metric="cpi_inflation",
                target_value=4.0,
                target_date=datetime(2021, 3, 31),
                source="Inflation Targeting Framework",
                category="monetary",
            ),
        }

    def validate_claim(
        self,
        claim_id: str,
        actual_data: Dict[str, pd.DataFrame],
    ) -> ClaimValidation:
        """
        Validate a specific claim against actual data.

        Args:
            claim_id: ID of the claim to validate
            actual_data: Dictionary of actual data

        Returns:
            ClaimValidation with detailed assessment
        """
        if claim_id not in self._claims:
            raise ValueError(f"Unknown claim: {claim_id}")

        claim = self._claims[claim_id]
        return self._validate_single_claim(claim, actual_data)

    def _validate_single_claim(
        self,
        claim: PoliticalClaim,
        actual_data: Dict[str, pd.DataFrame],
    ) -> ClaimValidation:
        """Validate a single claim."""
        evidence = []
        caveats = []

        # Get actual value if data available
        actual_value = None
        achievement_pct = None

        if claim.target_metric in actual_data:
            df = actual_data[claim.target_metric]

            if claim.target_date:
                # Get value closest to target date
                closest_date = df.index[
                    (df.index - claim.target_date).abs().argmin()
                ]
                actual_value = df.loc[closest_date, "value"]
                evidence.append(f"Actual value at {closest_date.date()}: {actual_value}")
            else:
                # Use latest value
                actual_value = df["value"].iloc[-1]
                evidence.append(f"Latest value: {actual_value}")

            # Calculate achievement percentage
            if claim.target_value and actual_value:
                if claim.target_value > 0:
                    achievement_pct = (actual_value / claim.target_value) * 100
                evidence.append(f"Achievement: {achievement_pct:.1f}% of target")

        # Determine status
        status = self._determine_status(claim, actual_value, achievement_pct)

        # Add caveats
        if claim.target_date and claim.target_date > datetime.now():
            caveats.append("Target date is in the future - assessment is preliminary")

        if actual_value is None:
            caveats.append("Could not find reliable data to validate this claim")

        # Methodology notes
        methodology = self._generate_methodology_notes(claim)

        # Confidence level
        confidence = self._calculate_confidence(claim, actual_data)

        return ClaimValidation(
            claim=claim,
            status=status,
            actual_value=actual_value,
            achievement_pct=achievement_pct,
            evidence=evidence,
            data_sources=[claim.target_metric],
            methodology_notes=methodology,
            confidence_level=confidence,
            caveats=caveats,
        )

    def _determine_status(
        self,
        claim: PoliticalClaim,
        actual_value: Optional[float],
        achievement_pct: Optional[float],
    ) -> str:
        """Determine the status of a claim."""
        if actual_value is None:
            return "cannot_verify"

        if claim.target_date and claim.target_date > datetime.now():
            return "ongoing"

        if achievement_pct is None:
            return "cannot_verify"

        if achievement_pct >= 90:
            return "met"
        elif achievement_pct >= 50:
            return "partially_met"
        else:
            return "not_met"

    def _generate_methodology_notes(self, claim: PoliticalClaim) -> str:
        """Generate methodology notes for validation."""
        notes = []

        notes.append(f"Claim made on: {claim.claim_date.date()}")
        notes.append(f"Target metric: {claim.target_metric}")

        if claim.target_value:
            notes.append(f"Target value: {claim.target_value}")
        if claim.target_date:
            notes.append(f"Target date: {claim.target_date.date()}")

        notes.append(
            "Validation uses official government data where available, "
            "supplemented by independent sources."
        )

        return " | ".join(notes)

    def _calculate_confidence(
        self,
        claim: PoliticalClaim,
        actual_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Calculate confidence level in the validation."""
        confidence = 1.0

        # Reduce confidence if data is missing
        if claim.target_metric not in actual_data:
            confidence *= 0.3

        # Reduce confidence for vague claims
        if claim.target_value is None:
            confidence *= 0.7

        # Reduce confidence for future targets
        if claim.target_date and claim.target_date > datetime.now():
            confidence *= 0.5

        return confidence

    def validate_all_claims(
        self,
        actual_data: Dict[str, pd.DataFrame],
        party: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, ClaimValidation]:
        """
        Validate all claims matching filters.

        Args:
            actual_data: Dictionary of actual data
            party: Filter by party (optional)
            category: Filter by category (optional)

        Returns:
            Dictionary of claim_id to ClaimValidation
        """
        results = {}

        for claim_id, claim in self._claims.items():
            # Apply filters
            if party and claim.party != party:
                continue
            if category and claim.category != category:
                continue

            try:
                results[claim_id] = self._validate_single_claim(claim, actual_data)
            except Exception as e:
                logger.warning(f"Failed to validate {claim_id}: {e}")

        return results

    def get_claim_scorecard(
        self,
        validations: Dict[str, ClaimValidation],
    ) -> Dict[str, Any]:
        """
        Generate scorecard from validations.

        Args:
            validations: Dictionary of validations

        Returns:
            Scorecard with summary statistics
        """
        if not validations:
            return {"error": "No validations provided"}

        # Count by status
        status_counts = {
            "met": 0,
            "partially_met": 0,
            "not_met": 0,
            "ongoing": 0,
            "cannot_verify": 0,
        }

        for v in validations.values():
            if v.status in status_counts:
                status_counts[v.status] += 1

        # Calculate achievement rate
        verifiable = status_counts["met"] + status_counts["partially_met"] + status_counts["not_met"]
        if verifiable > 0:
            met_rate = status_counts["met"] / verifiable * 100
            partially_met_rate = status_counts["partially_met"] / verifiable * 100
        else:
            met_rate = 0
            partially_met_rate = 0

        # Group by category
        by_category = {}
        for v in validations.values():
            cat = v.claim.category
            if cat not in by_category:
                by_category[cat] = {"met": 0, "not_met": 0, "total": 0}
            by_category[cat]["total"] += 1
            if v.status == "met":
                by_category[cat]["met"] += 1
            elif v.status == "not_met":
                by_category[cat]["not_met"] += 1

        return {
            "total_claims": len(validations),
            "status_breakdown": status_counts,
            "verifiable_claims": verifiable,
            "met_rate_pct": met_rate,
            "partially_met_rate_pct": partially_met_rate,
            "by_category": by_category,
            "average_confidence": np.mean([v.confidence_level for v in validations.values()]),
        }

    def compare_party_track_records(
        self,
        actual_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Compare track records of different parties.

        Args:
            actual_data: Dictionary of actual data

        Returns:
            Comparison of party track records
        """
        party_results = {}

        for party in ["BJP", "INC", "International", "RBI"]:
            validations = self.validate_all_claims(actual_data, party=party)
            if validations:
                party_results[party] = self.get_claim_scorecard(validations)

        return party_results

    def add_claim(
        self,
        claim: PoliticalClaim,
    ) -> None:
        """Add a new claim to track."""
        self._claims[claim.id] = claim
        logger.info(f"Added claim: {claim.id}")

    def get_claims_by_category(
        self,
        category: str,
    ) -> List[PoliticalClaim]:
        """Get all claims in a category."""
        return [
            c for c in self._claims.values()
            if c.category == category
        ]

    def get_unfulfilled_promises(
        self,
        actual_data: Dict[str, pd.DataFrame],
        threshold: float = 50.0,
    ) -> List[ClaimValidation]:
        """
        Get promises that were not fulfilled.

        Args:
            actual_data: Dictionary of actual data
            threshold: Achievement percentage below which considered unfulfilled

        Returns:
            List of unfulfilled promise validations
        """
        all_validations = self.validate_all_claims(actual_data)

        unfulfilled = [
            v for v in all_validations.values()
            if v.achievement_pct is not None and v.achievement_pct < threshold
        ]

        return sorted(unfulfilled, key=lambda x: x.achievement_pct or 0)

    def generate_accountability_report(
        self,
        actual_data: Dict[str, pd.DataFrame],
        party: str,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive accountability report for a party.

        Args:
            actual_data: Dictionary of actual data
            party: Party to analyze

        Returns:
            Comprehensive accountability report
        """
        validations = self.validate_all_claims(actual_data, party=party)
        scorecard = self.get_claim_scorecard(validations)

        # Get notable successes and failures
        met_claims = [v for v in validations.values() if v.status == "met"]
        failed_claims = [v for v in validations.values() if v.status == "not_met"]

        return {
            "party": party,
            "summary": scorecard,
            "notable_successes": [
                {
                    "claim": v.claim.claim_text,
                    "achievement": f"{v.achievement_pct:.1f}%",
                }
                for v in met_claims[:5]
            ],
            "notable_failures": [
                {
                    "claim": v.claim.claim_text,
                    "achievement": f"{v.achievement_pct:.1f}%" if v.achievement_pct else "N/A",
                }
                for v in failed_claims[:5]
            ],
            "methodology": (
                "Claims validated using official government data, "
                "supplemented by independent sources where available. "
                "Achievement percentages calculated as actual/target * 100."
            ),
            "caveats": [
                "Some claims are vague and difficult to measure precisely",
                "Data availability varies by metric",
                "External factors may affect outcomes",
            ],
        }
