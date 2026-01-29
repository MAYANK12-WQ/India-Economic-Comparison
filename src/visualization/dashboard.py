"""
Interactive Dashboard Builder using Dash/Plotly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from dash import Dash, html, dcc, callback, Output, Input
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not available - dashboard features limited")


class DashboardBuilder:
    """
    Builds interactive dashboards for economic comparison.
    """

    def __init__(self, title: str = "India Economic Comparison Dashboard"):
        self.title = title
        self.app: Optional[Dash] = None

        if DASH_AVAILABLE:
            self._init_app()

    def _init_app(self) -> None:
        """Initialize Dash application."""
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title=self.title,
        )

        self.app.layout = self._create_layout()
        self._register_callbacks()

    def _create_layout(self) -> Any:
        """Create dashboard layout."""
        if not DASH_AVAILABLE:
            return None

        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center my-4"),
                    html.P(
                        "Comprehensive economic comparison: UPA (2004-2014) vs NDA (2014-2024)",
                        className="text-center text-muted"
                    ),
                ])
            ]),

            # Period selector
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Comparison"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Indicator"),
                                    dcc.Dropdown(
                                        id="indicator-selector",
                                        options=[
                                            {"label": "GDP Growth Rate", "value": "gdp_growth_rate"},
                                            {"label": "Inflation (CPI)", "value": "cpi_inflation"},
                                            {"label": "Fiscal Deficit", "value": "fiscal_deficit"},
                                            {"label": "Forex Reserves", "value": "forex_reserves"},
                                            {"label": "Unemployment Rate", "value": "unemployment_rate"},
                                            {"label": "FDI Inflows", "value": "fdi_inflows"},
                                        ],
                                        value="gdp_growth_rate",
                                    ),
                                ], md=6),
                                dbc.Col([
                                    dbc.Label("Chart Type"),
                                    dcc.Dropdown(
                                        id="chart-type-selector",
                                        options=[
                                            {"label": "Time Series", "value": "line"},
                                            {"label": "Bar Comparison", "value": "bar"},
                                            {"label": "Distribution", "value": "box"},
                                        ],
                                        value="line",
                                    ),
                                ], md=6),
                            ]),
                        ]),
                    ], className="mb-4"),
                ])
            ]),

            # Main chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Comparison Chart"),
                        dbc.CardBody([
                            dcc.Graph(id="main-chart"),
                        ]),
                    ]),
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Summary Statistics"),
                        dbc.CardBody(id="summary-stats"),
                    ]),
                ], md=4),
            ], className="mb-4"),

            # Period comparison cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("UPA Era (2004-2014)", className="bg-primary text-white"),
                        dbc.CardBody(id="upa-stats"),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("NDA Era (2014-2024)", className="bg-warning"),
                        dbc.CardBody(id="nda-stats"),
                    ]),
                ], md=6),
            ], className="mb-4"),

            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Data sources: MOSPI, RBI, World Bank, IMF | "
                        "Disclaimer: For educational purposes only",
                        className="text-center text-muted small"
                    ),
                ])
            ]),

        ], fluid=True)

    def _register_callbacks(self) -> None:
        """Register dashboard callbacks."""
        if not DASH_AVAILABLE or self.app is None:
            return

        @self.app.callback(
            [
                Output("main-chart", "figure"),
                Output("summary-stats", "children"),
                Output("upa-stats", "children"),
                Output("nda-stats", "children"),
            ],
            [
                Input("indicator-selector", "value"),
                Input("chart-type-selector", "value"),
            ]
        )
        def update_dashboard(indicator: str, chart_type: str):
            """Update dashboard based on selections."""
            import plotly.graph_objects as go
            import numpy as np

            # Sample data (in production, fetch from data pipeline)
            data = self._get_sample_data(indicator)

            # Create chart
            if chart_type == "line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(data.keys()),
                    y=list(data.values()),
                    mode="lines+markers",
                    name=indicator,
                ))

                # Add period shading
                fig.add_vrect(
                    x0="2004-05", x1="2013-14",
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line_width=0,
                    annotation_text="UPA",
                )
                fig.add_vrect(
                    x0="2014-15", x1="2023-24",
                    fillcolor="rgba(255, 165, 0, 0.1)",
                    line_width=0,
                    annotation_text="NDA",
                )

            elif chart_type == "bar":
                years = list(data.keys())
                values = list(data.values())
                colors = ["blue" if y < "2014-15" else "orange" for y in years]

                fig = go.Figure(data=[
                    go.Bar(x=years, y=values, marker_color=colors)
                ])

            else:  # box
                upa_values = [v for k, v in data.items() if k < "2014-15"]
                nda_values = [v for k, v in data.items() if k >= "2014-15"]

                fig = go.Figure()
                fig.add_trace(go.Box(y=upa_values, name="UPA (2004-14)"))
                fig.add_trace(go.Box(y=nda_values, name="NDA (2014-24)"))

            fig.update_layout(
                title=f"{indicator.replace('_', ' ').title()} Over Time",
                template="plotly_white",
                height=400,
            )

            # Summary stats
            all_values = list(data.values())
            upa_values = [v for k, v in data.items() if k < "2014-15"]
            nda_values = [v for k, v in data.items() if k >= "2014-15"]

            summary = html.Div([
                html.P(f"Overall Average: {np.mean(all_values):.2f}"),
                html.P(f"Overall Std Dev: {np.std(all_values):.2f}"),
                html.P(f"Min: {min(all_values):.2f}"),
                html.P(f"Max: {max(all_values):.2f}"),
            ])

            upa_stats = html.Div([
                html.H4(f"{np.mean(upa_values):.2f}", className="text-primary"),
                html.P("Average"),
                html.P(f"Std Dev: {np.std(upa_values):.2f}"),
            ])

            nda_stats = html.Div([
                html.H4(f"{np.mean(nda_values):.2f}", className="text-warning"),
                html.P("Average"),
                html.P(f"Std Dev: {np.std(nda_values):.2f}"),
            ])

            return fig, summary, upa_stats, nda_stats

    def _get_sample_data(self, indicator: str) -> Dict[str, float]:
        """Get sample data for an indicator."""
        sample_data = {
            "gdp_growth_rate": {
                "2004-05": 7.1, "2005-06": 9.5, "2006-07": 9.6, "2007-08": 9.3,
                "2008-09": 6.7, "2009-10": 8.6, "2010-11": 8.9, "2011-12": 6.7,
                "2012-13": 5.5, "2013-14": 6.4, "2014-15": 7.4, "2015-16": 8.0,
                "2016-17": 8.3, "2017-18": 6.8, "2018-19": 6.5, "2019-20": 3.7,
                "2020-21": -6.6, "2021-22": 8.7, "2022-23": 7.2, "2023-24": 7.6,
            },
            "cpi_inflation": {
                "2004-05": 3.8, "2005-06": 4.4, "2006-07": 6.5, "2007-08": 6.2,
                "2008-09": 9.1, "2009-10": 12.4, "2010-11": 10.4, "2011-12": 8.9,
                "2012-13": 10.2, "2013-14": 9.4, "2014-15": 5.9, "2015-16": 4.9,
                "2016-17": 4.5, "2017-18": 3.6, "2018-19": 3.4, "2019-20": 4.8,
                "2020-21": 6.2, "2021-22": 5.5, "2022-23": 6.7, "2023-24": 5.4,
            },
            "fiscal_deficit": {
                "2004-05": 4.0, "2005-06": 4.1, "2006-07": 3.5, "2007-08": 2.5,
                "2008-09": 6.0, "2009-10": 6.5, "2010-11": 4.8, "2011-12": 5.9,
                "2012-13": 4.9, "2013-14": 4.5, "2014-15": 4.1, "2015-16": 3.9,
                "2016-17": 3.5, "2017-18": 3.5, "2018-19": 3.4, "2019-20": 4.6,
                "2020-21": 9.2, "2021-22": 6.7, "2022-23": 6.4, "2023-24": 5.9,
            },
            "forex_reserves": {
                "2004-05": 141, "2005-06": 151, "2006-07": 199, "2007-08": 310,
                "2008-09": 252, "2009-10": 279, "2010-11": 305, "2011-12": 294,
                "2012-13": 292, "2013-14": 304, "2014-15": 341, "2015-16": 360,
                "2016-17": 370, "2017-18": 424, "2018-19": 412, "2019-20": 478,
                "2020-21": 577, "2021-22": 607, "2022-23": 578, "2023-24": 645,
            },
            "unemployment_rate": {
                "2004-05": 5.3, "2005-06": 5.2, "2006-07": 5.1, "2007-08": 4.9,
                "2008-09": 5.0, "2009-10": 4.8, "2010-11": 4.7, "2011-12": 4.5,
                "2012-13": 4.7, "2013-14": 4.9, "2014-15": 5.0, "2015-16": 5.2,
                "2016-17": 5.4, "2017-18": 6.0, "2018-19": 5.8, "2019-20": 5.3,
                "2020-21": 8.0, "2021-22": 5.8, "2022-23": 5.4, "2023-24": 5.2,
            },
            "fdi_inflows": {
                "2004-05": 6.0, "2005-06": 8.9, "2006-07": 22.8, "2007-08": 34.8,
                "2008-09": 41.9, "2009-10": 37.7, "2010-11": 34.8, "2011-12": 46.5,
                "2012-13": 34.3, "2013-14": 36.0, "2014-15": 45.1, "2015-16": 55.5,
                "2016-17": 60.2, "2017-18": 61.0, "2018-19": 62.0, "2019-20": 74.4,
                "2020-21": 81.7, "2021-22": 83.6, "2022-23": 71.4, "2023-24": 70.9,
            },
        }

        return sample_data.get(indicator, sample_data["gdp_growth_rate"])

    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard server."""
        if not DASH_AVAILABLE or self.app is None:
            logger.error("Dash not available. Install with: pip install dash dash-bootstrap-components")
            return

        self.app.run_server(host=host, port=port, debug=debug)
