"""
Chart generation for economic comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try importing visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - visualization features limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    title: str
    width: int = 800
    height: int = 500
    theme: str = "plotly_white"
    show_uncertainty: bool = True
    period_colors: Dict[str, str] = None

    def __post_init__(self):
        if self.period_colors is None:
            self.period_colors = {
                "UPA": "#1f77b4",  # Blue
                "NDA": "#ff7f0e",  # Orange
                "Overlap": "#2ca02c",  # Green
            }


class ComparisonChart:
    """
    Generate comparison charts between two periods.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig(title="Economic Comparison")

    def bar_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
    ) -> Any:
        """
        Create side-by-side bar comparison.

        Args:
            metrics: Dict of metric_name -> {"UPA": value, "NDA": value}
            title: Chart title

        Returns:
            Plotly figure or matplotlib axes
        """
        if not PLOTLY_AVAILABLE:
            return self._bar_comparison_matplotlib(metrics, title)

        metric_names = list(metrics.keys())
        upa_values = [metrics[m].get("UPA", 0) for m in metric_names]
        nda_values = [metrics[m].get("NDA", 0) for m in metric_names]

        fig = go.Figure(data=[
            go.Bar(
                name="UPA (2004-14)",
                x=metric_names,
                y=upa_values,
                marker_color=self.config.period_colors["UPA"],
            ),
            go.Bar(
                name="NDA (2014-24)",
                x=metric_names,
                y=nda_values,
                marker_color=self.config.period_colors["NDA"],
            ),
        ])

        fig.update_layout(
            title=title or self.config.title,
            barmode="group",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def _bar_comparison_matplotlib(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: Optional[str],
    ) -> Any:
        """Matplotlib fallback for bar comparison."""
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "No visualization library available"}

        metric_names = list(metrics.keys())
        upa_values = [metrics[m].get("UPA", 0) for m in metric_names]
        nda_values = [metrics[m].get("NDA", 0) for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, upa_values, width, label="UPA (2004-14)")
        ax.bar(x + width/2, nda_values, width, label="NDA (2014-24)")

        ax.set_ylabel("Value")
        ax.set_title(title or self.config.title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        return fig

    def waterfall_decomposition(
        self,
        components: Dict[str, float],
        title: str = "Growth Decomposition",
    ) -> Any:
        """
        Create waterfall chart for decomposition.

        Args:
            components: Dict of component_name -> contribution
            title: Chart title

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for waterfall charts"}

        names = list(components.keys())
        values = list(components.values())

        # Determine measure type
        measures = []
        for i, v in enumerate(values):
            if i == 0:
                measures.append("absolute")
            elif i == len(values) - 1:
                measures.append("total")
            else:
                measures.append("relative")

        fig = go.Figure(go.Waterfall(
            name="Decomposition",
            orientation="v",
            measure=measures,
            x=names,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
        ))

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig


class TimeSeriesChart:
    """
    Generate time series charts with period annotations.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig(title="Time Series")

    def line_with_periods(
        self,
        data: pd.DataFrame,
        column: str = "value",
        periods: Optional[Dict[str, Tuple[str, str]]] = None,
        title: Optional[str] = None,
    ) -> Any:
        """
        Create line chart with shaded period backgrounds.

        Args:
            data: DataFrame with datetime index
            column: Column to plot
            periods: Dict of period_name -> (start_date, end_date)
            title: Chart title

        Returns:
            Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            return self._line_matplotlib(data, column, title)

        if periods is None:
            periods = {
                "UPA": ("2004-05-22", "2014-05-26"),
                "NDA": ("2014-05-26", "2024-06-04"),
            }

        fig = go.Figure()

        # Add main line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode="lines",
            name=column,
            line=dict(color="#1f77b4", width=2),
        ))

        # Add period shading
        colors = {"UPA": "rgba(31, 119, 180, 0.1)", "NDA": "rgba(255, 127, 14, 0.1)"}
        for period_name, (start, end) in periods.items():
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=colors.get(period_name, "rgba(128, 128, 128, 0.1)"),
                layer="below",
                line_width=0,
                annotation_text=period_name,
                annotation_position="top left",
            )

        fig.update_layout(
            title=title or self.config.title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            xaxis_title="Date",
            yaxis_title=column,
        )

        return fig

    def _line_matplotlib(
        self,
        data: pd.DataFrame,
        column: str,
        title: Optional[str],
    ) -> Any:
        """Matplotlib fallback for line chart."""
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "No visualization library available"}

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data[column])
        ax.set_title(title or self.config.title)
        ax.set_xlabel("Date")
        ax.set_ylabel(column)

        # Add period shading
        ax.axvspan("2004-05-22", "2014-05-26", alpha=0.1, color="blue", label="UPA")
        ax.axvspan("2014-05-26", "2024-06-04", alpha=0.1, color="orange", label="NDA")

        ax.legend()
        plt.tight_layout()
        return fig

    def dual_axis(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        col1: str = "value",
        col2: str = "value",
        label1: str = "Series 1",
        label2: str = "Series 2",
        title: Optional[str] = None,
    ) -> Any:
        """
        Create dual-axis chart for comparing different scale metrics.
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for dual-axis charts"}

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=data1.index, y=data1[col1], name=label1, line=dict(color="#1f77b4")),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=data2.index, y=data2[col2], name=label2, line=dict(color="#ff7f0e")),
            secondary_y=True,
        )

        fig.update_layout(
            title=title or "Dual Axis Comparison",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        fig.update_yaxes(title_text=label1, secondary_y=False)
        fig.update_yaxes(title_text=label2, secondary_y=True)

        return fig


class UncertaintyChart:
    """
    Charts that visualize uncertainty and confidence intervals.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig(title="Uncertainty Visualization")

    def confidence_band(
        self,
        data: pd.DataFrame,
        point_col: str = "value",
        lower_col: str = "ci_lower",
        upper_col: str = "ci_upper",
        title: Optional[str] = None,
    ) -> Any:
        """
        Create line chart with confidence band.
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for confidence bands"}

        fig = go.Figure()

        # Add confidence band
        fig.add_trace(go.Scatter(
            x=list(data.index) + list(data.index[::-1]),
            y=list(data[upper_col]) + list(data[lower_col][::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="95% CI",
        ))

        # Add point estimate line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[point_col],
            mode="lines",
            name="Point Estimate",
            line=dict(color="#1f77b4", width=2),
        ))

        fig.update_layout(
            title=title or "Estimate with Confidence Interval",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def distribution_comparison(
        self,
        distributions: Dict[str, np.ndarray],
        title: str = "Distribution Comparison",
    ) -> Any:
        """
        Compare distributions across periods.
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for distribution plots"}

        fig = go.Figure()

        for name, values in distributions.items():
            fig.add_trace(go.Violin(
                y=values,
                name=name,
                box_visible=True,
                meanline_visible=True,
            ))

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def forest_plot(
        self,
        estimates: List[Dict[str, Any]],
        title: str = "Forest Plot",
    ) -> Any:
        """
        Create forest plot for multiple estimates.

        Args:
            estimates: List of {"name": str, "point": float, "ci_lower": float, "ci_upper": float}
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for forest plots"}

        fig = go.Figure()

        names = [e["name"] for e in estimates]
        points = [e["point"] for e in estimates]
        ci_lowers = [e["ci_lower"] for e in estimates]
        ci_uppers = [e["ci_upper"] for e in estimates]

        # Add confidence intervals as error bars
        fig.add_trace(go.Scatter(
            x=points,
            y=names,
            mode="markers",
            marker=dict(size=10, color="#1f77b4"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - p for u, p in zip(ci_uppers, points)],
                arrayminus=[p - l for p, l in zip(points, ci_lowers)],
            ),
            name="Estimates",
        ))

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            xaxis_title="Effect Size",
        )

        return fig


class HeatmapChart:
    """
    Heatmap visualizations for correlation and comparison matrices.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig(title="Heatmap")

    def correlation_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
    ) -> Any:
        """
        Create correlation heatmap.
        """
        if not PLOTLY_AVAILABLE:
            return self._correlation_matplotlib(data, title)

        corr = data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def _correlation_matplotlib(
        self,
        data: pd.DataFrame,
        title: str,
    ) -> Any:
        """Matplotlib fallback for correlation heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "No visualization library available"}

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap="RdBu", center=0, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def period_indicator_matrix(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Period-Indicator Matrix",
    ) -> Any:
        """
        Create heatmap comparing indicators across periods.

        Args:
            data: Dict of indicator -> {"UPA": value, "NDA": value}
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for heatmaps"}

        indicators = list(data.keys())
        periods = ["UPA", "NDA"]

        z = [[data[ind].get(p, 0) for p in periods] for ind in indicators]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=periods,
            y=indicators,
            colorscale="RdYlGn",
            text=np.round(z, 2),
            texttemplate="%{text}",
        ))

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig


class RadarChart:
    """
    Radar/spider charts for multi-dimensional comparison.
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig(title="Radar Chart")

    def multi_dimension_comparison(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Multi-Dimensional Comparison",
    ) -> Any:
        """
        Create radar chart comparing multiple dimensions.

        Args:
            data: Dict of period -> {metric: normalized_value}
                  Values should be normalized to 0-100 scale
        """
        if not PLOTLY_AVAILABLE:
            return {"error": "Plotly required for radar charts"}

        fig = go.Figure()

        for period, metrics in data.items():
            categories = list(metrics.keys())
            values = list(metrics.values())
            # Close the polygon
            values.append(values[0])
            categories.append(categories[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=period,
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def scorecard_radar(
        self,
        scores: Dict[str, float],
        title: str = "Performance Scorecard",
    ) -> Any:
        """
        Create single-period radar scorecard.
        """
        return self.multi_dimension_comparison(
            {"Performance": scores},
            title=title,
        )
