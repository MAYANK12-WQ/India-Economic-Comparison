"""
Main Data Pipeline orchestration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    source_type: str
    url: str
    reliability_score: float
    update_frequency: str
    lag_days: int = 0
    indicators: List[str] = field(default_factory=list)
    authentication: Optional[Dict[str, str]] = None


@dataclass
class DataPoint:
    """A single data point with metadata."""
    indicator: str
    value: float
    date: datetime
    source: str
    reliability_score: float
    revision_number: int = 0
    methodology_version: str = ""
    notes: str = ""


@dataclass
class TimeSeries:
    """A time series with full provenance."""
    indicator: str
    data: pd.DataFrame
    source: str
    reliability_score: float
    methodology_changes: List[Dict[str, Any]] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator": self.indicator,
            "data": self.data.to_dict(orient="records"),
            "source": self.source,
            "reliability_score": self.reliability_score,
            "methodology_changes": self.methodology_changes,
            "quality_flags": self.quality_flags,
        }


class DataPipeline:
    """
    Main data pipeline for fetching, cleaning, and validating economic data.

    Features:
    - Multi-source data fetching with retry logic
    - Automatic caching with configurable TTL
    - Data validation and quality scoring
    - Time alignment across different frequencies
    - Base year splicing for GDP series
    - Seasonal adjustment
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.config = self._load_config(config_path)
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._sources: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._data_store: Dict[str, TimeSeries] = {}

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/settings.yaml")

        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def register_source(self, name: str, source: Any) -> None:
        """Register a data source."""
        self._sources[name] = source
        logger.info(f"Registered data source: {name}")

    def fetch_indicator(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> TimeSeries:
        """
        Fetch an indicator from configured sources.

        Args:
            indicator: The indicator to fetch (e.g., 'gdp_growth_rate')
            start_date: Start of the date range
            end_date: End of the date range
            sources: List of sources to query (defaults to all)
            force_refresh: Bypass cache if True

        Returns:
            TimeSeries with data and metadata
        """
        cache_key = self._get_cache_key(indicator, start_date, end_date, sources)

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - cached["timestamp"] < timedelta(hours=24):
                logger.info(f"Cache hit for {indicator}")
                return cached["data"]

        # Fetch from sources
        results = []
        sources_to_query = sources or list(self._sources.keys())

        for source_name in sources_to_query:
            if source_name not in self._sources:
                continue

            source = self._sources[source_name]
            try:
                data = source.fetch(indicator, start_date, end_date)
                if data is not None:
                    results.append({
                        "source": source_name,
                        "data": data,
                        "reliability": source.reliability_score,
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator} from {source_name}: {e}")

        if not results:
            raise ValueError(f"No data found for indicator: {indicator}")

        # Merge results with reliability weighting
        merged = self._merge_sources(results, indicator)

        # Cache result
        self._cache[cache_key] = {
            "data": merged,
            "timestamp": datetime.now(),
        }

        return merged

    def _merge_sources(
        self,
        results: List[Dict[str, Any]],
        indicator: str,
    ) -> TimeSeries:
        """Merge data from multiple sources with reliability weighting."""
        if len(results) == 1:
            r = results[0]
            return TimeSeries(
                indicator=indicator,
                data=r["data"],
                source=r["source"],
                reliability_score=r["reliability"],
            )

        # Sort by reliability
        results.sort(key=lambda x: x["reliability"], reverse=True)

        # Use highest reliability source as base
        base = results[0]
        merged_df = base["data"].copy()

        # Flag discrepancies
        quality_flags = []
        for other in results[1:]:
            other_df = other["data"]

            # Check for significant discrepancies
            common_dates = merged_df.index.intersection(other_df.index)
            if len(common_dates) > 0:
                discrepancy = (
                    merged_df.loc[common_dates, "value"] -
                    other_df.loc[common_dates, "value"]
                ).abs().mean()

                if discrepancy > merged_df["value"].std() * 0.1:
                    quality_flags.append(
                        f"Discrepancy with {other['source']}: {discrepancy:.2f}"
                    )

        # Calculate weighted average reliability
        total_weight = sum(r["reliability"] for r in results)
        avg_reliability = sum(
            r["reliability"] ** 2 for r in results
        ) / total_weight

        return TimeSeries(
            indicator=indicator,
            data=merged_df,
            source=f"merged({', '.join(r['source'] for r in results)})",
            reliability_score=avg_reliability,
            quality_flags=quality_flags,
        )

    def _get_cache_key(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]],
    ) -> str:
        """Generate a cache key."""
        key_data = {
            "indicator": indicator,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "sources": sorted(sources) if sources else None,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def align_frequencies(
        self,
        series_dict: Dict[str, TimeSeries],
        target_frequency: str = "Q",
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple time series to a common frequency.

        Args:
            series_dict: Dictionary of indicator name to TimeSeries
            target_frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            Dictionary of aligned DataFrames
        """
        aligned = {}

        for name, ts in series_dict.items():
            df = ts.data.copy()

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Resample to target frequency
            if target_frequency == "Q":
                aligned[name] = df.resample("Q").mean()
            elif target_frequency == "M":
                aligned[name] = df.resample("M").mean()
            elif target_frequency == "Y":
                aligned[name] = df.resample("Y").mean()
            else:
                aligned[name] = df

        return aligned

    def splice_base_years(
        self,
        old_series: pd.DataFrame,
        new_series: pd.DataFrame,
        overlap_start: datetime,
        overlap_end: datetime,
        method: str = "ratio_splicing",
    ) -> pd.DataFrame:
        """
        Splice GDP series with different base years.

        Args:
            old_series: GDP with old base year (2004-05)
            new_series: GDP with new base year (2011-12)
            overlap_start: Start of overlap period
            overlap_end: End of overlap period
            method: Splicing method ('ratio_splicing', 'growth_rate', 'interpolation')

        Returns:
            Spliced continuous series
        """
        if method == "ratio_splicing":
            # Calculate ratio during overlap period
            overlap_old = old_series.loc[overlap_start:overlap_end]
            overlap_new = new_series.loc[overlap_start:overlap_end]

            ratio = (overlap_new["value"] / overlap_old["value"]).mean()

            # Adjust old series
            adjusted_old = old_series.copy()
            adjusted_old["value"] = adjusted_old["value"] * ratio

            # Combine series
            spliced = pd.concat([
                adjusted_old.loc[:overlap_start - timedelta(days=1)],
                new_series.loc[overlap_start:],
            ])

        elif method == "growth_rate":
            # Use growth rates to back-cast
            new_growth = new_series["value"].pct_change()

            # Back-cast using old series growth rates
            old_growth = old_series["value"].pct_change()

            # Start from new series first available point
            start_value = new_series["value"].iloc[0]
            back_casted = [start_value]

            for i in range(len(old_series) - 1, -1, -1):
                if old_series.index[i] < new_series.index[0]:
                    prev_value = back_casted[-1] / (1 + old_growth.iloc[i])
                    back_casted.append(prev_value)

            back_casted.reverse()
            spliced = pd.DataFrame({
                "value": back_casted + new_series["value"].tolist()[1:],
            }, index=list(old_series.index) + list(new_series.index[1:]))

        else:
            raise ValueError(f"Unknown splicing method: {method}")

        return spliced

    def get_all_indicators(
        self,
        start_date: datetime,
        end_date: datetime,
        indicator_list: Optional[List[str]] = None,
    ) -> Dict[str, TimeSeries]:
        """
        Fetch all configured indicators.

        Returns:
            Dictionary of indicator name to TimeSeries
        """
        if indicator_list is None:
            indicator_list = []
            for category in self.config.get("indicators", {}).values():
                indicator_list.extend(category)

        results = {}
        for indicator in indicator_list:
            try:
                results[indicator] = self.fetch_indicator(
                    indicator, start_date, end_date
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator}: {e}")

        return results

    def export_dataset(
        self,
        output_path: Path,
        format: str = "parquet",
    ) -> None:
        """Export the complete dataset."""
        all_data = []

        for name, ts in self._data_store.items():
            df = ts.data.copy()
            df["indicator"] = name
            df["source"] = ts.source
            df["reliability_score"] = ts.reliability_score
            all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            if format == "parquet":
                combined.to_parquet(output_path)
            elif format == "csv":
                combined.to_csv(output_path, index=False)
            elif format == "json":
                combined.to_json(output_path, orient="records", date_format="iso")

            logger.info(f"Exported dataset to {output_path}")

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a data quality report."""
        report = {
            "total_indicators": len(self._data_store),
            "sources_used": set(),
            "date_range": {"min": None, "max": None},
            "quality_issues": [],
            "coverage": {},
        }

        for name, ts in self._data_store.items():
            report["sources_used"].add(ts.source)

            if ts.quality_flags:
                report["quality_issues"].extend([
                    {"indicator": name, "flag": flag}
                    for flag in ts.quality_flags
                ])

            # Check date coverage
            if len(ts.data) > 0:
                min_date = ts.data.index.min()
                max_date = ts.data.index.max()

                if report["date_range"]["min"] is None or min_date < report["date_range"]["min"]:
                    report["date_range"]["min"] = min_date
                if report["date_range"]["max"] is None or max_date > report["date_range"]["max"]:
                    report["date_range"]["max"] = max_date

                report["coverage"][name] = {
                    "start": min_date,
                    "end": max_date,
                    "completeness": 1 - ts.data["value"].isna().mean(),
                }

        report["sources_used"] = list(report["sources_used"])
        return report
