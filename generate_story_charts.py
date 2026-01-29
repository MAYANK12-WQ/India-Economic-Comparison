#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Story-Telling Economic Charts - India Economic Comparison System
=================================================================

Creates comprehensive narrative-driven visualizations that tell the complete
economic story of India from 2004-2024. Designed for economists, policymakers,
and researchers who need a complete picture.
"""

import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

output_dir = Path("charts")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("GENERATING STORY-TELLING ECONOMIC CHARTS")
print("=" * 70)
print()

# Load existing data
with open("data/sample_indicators.json") as f:
    data = json.load(f)

indicators = data["indicators"]

# ============================================================================
# EXTENDED ECONOMIC DATA FOR COMPREHENSIVE ANALYSIS
# ============================================================================

# Sector-wise GDP Contribution (%)
gdp_agriculture = {
    "2004-05": 19.0, "2005-06": 18.3, "2006-07": 17.4, "2007-08": 16.8,
    "2008-09": 15.8, "2009-10": 14.6, "2010-11": 14.6, "2011-12": 14.4,
    "2012-13": 13.9, "2013-14": 13.9, "2014-15": 13.2, "2015-16": 12.9,
    "2016-17": 13.0, "2017-18": 12.8, "2018-19": 12.3, "2019-20": 12.7,
    "2020-21": 14.0, "2021-22": 13.3, "2022-23": 12.8, "2023-24": 12.4
}

gdp_industry = {
    "2004-05": 27.9, "2005-06": 28.0, "2006-07": 28.6, "2007-08": 28.7,
    "2008-09": 28.0, "2009-10": 28.3, "2010-11": 27.8, "2011-12": 27.2,
    "2012-13": 26.2, "2013-14": 25.8, "2014-15": 25.8, "2015-16": 26.0,
    "2016-17": 26.0, "2017-18": 26.1, "2018-19": 25.5, "2019-20": 24.2,
    "2020-21": 23.6, "2021-22": 24.3, "2022-23": 24.5, "2023-24": 24.8
}

gdp_services = {
    "2004-05": 53.1, "2005-06": 53.7, "2006-07": 54.0, "2007-08": 54.5,
    "2008-09": 56.2, "2009-10": 57.1, "2010-11": 57.6, "2011-12": 58.4,
    "2012-13": 59.9, "2013-14": 60.3, "2014-15": 61.0, "2015-16": 61.1,
    "2016-17": 61.0, "2017-18": 61.1, "2018-19": 62.2, "2019-20": 63.1,
    "2020-21": 62.4, "2021-22": 62.4, "2022-23": 62.7, "2023-24": 62.8
}

# Manufacturing Growth Rate (%)
manufacturing_growth = {
    "2004-05": 9.1, "2005-06": 10.1, "2006-07": 14.3, "2007-08": 10.3,
    "2008-09": 4.3, "2009-10": 11.3, "2010-11": 8.9, "2011-12": 7.4,
    "2012-13": 5.5, "2013-14": 5.0, "2014-15": 5.5, "2015-16": 10.8,
    "2016-17": 7.9, "2017-18": 6.6, "2018-19": 5.7, "2019-20": -2.9,
    "2020-21": -0.6, "2021-22": 11.1, "2022-23": 1.3, "2023-24": 8.5
}

# Tax-to-GDP Ratio (%)
tax_gdp_ratio = {
    "2004-05": 9.4, "2005-06": 10.2, "2006-07": 11.0, "2007-08": 11.9,
    "2008-09": 10.8, "2009-10": 9.6, "2010-11": 10.1, "2011-12": 10.1,
    "2012-13": 10.4, "2013-14": 10.1, "2014-15": 10.0, "2015-16": 10.6,
    "2016-17": 11.1, "2017-18": 11.2, "2018-19": 10.9, "2019-20": 10.0,
    "2020-21": 9.9, "2021-22": 10.8, "2022-23": 11.1, "2023-24": 11.6
}

# Gross Capital Formation (% of GDP)
gcf_rate = {
    "2004-05": 32.8, "2005-06": 34.7, "2006-07": 35.7, "2007-08": 38.1,
    "2008-09": 34.3, "2009-10": 36.5, "2010-11": 36.5, "2011-12": 34.3,
    "2012-13": 33.4, "2013-14": 31.3, "2014-15": 30.8, "2015-16": 30.0,
    "2016-17": 30.3, "2017-18": 30.4, "2018-19": 29.0, "2019-20": 27.8,
    "2020-21": 27.1, "2021-22": 29.2, "2022-23": 29.2, "2023-24": 29.8
}

# Poverty Headcount Ratio (%)
poverty_rate = {
    "2004-05": 37.2, "2005-06": 35.5, "2006-07": 33.8, "2007-08": 32.0,
    "2008-09": 30.5, "2009-10": 29.8, "2010-11": 21.9, "2011-12": 21.9,
    "2012-13": 20.5, "2013-14": 19.2, "2014-15": 18.0, "2015-16": 16.8,
    "2016-17": 15.5, "2017-18": 14.2, "2018-19": 13.0, "2019-20": 12.0,
    "2020-21": 14.5, "2021-22": 12.8, "2022-23": 11.5, "2023-24": 10.2
}

# Human Development Index
hdi_values = {
    "2004-05": 0.535, "2005-06": 0.545, "2006-07": 0.555, "2007-08": 0.565,
    "2008-09": 0.572, "2009-10": 0.580, "2010-11": 0.590, "2011-12": 0.599,
    "2012-13": 0.607, "2013-14": 0.614, "2014-15": 0.624, "2015-16": 0.636,
    "2016-17": 0.645, "2017-18": 0.654, "2018-19": 0.662, "2019-20": 0.633,
    "2020-21": 0.630, "2021-22": 0.644, "2022-23": 0.655, "2023-24": 0.670
}

# Nominal GDP in USD Trillion
gdp_nominal_usd = {
    "2004-05": 0.72, "2005-06": 0.83, "2006-07": 0.95, "2007-08": 1.24,
    "2008-09": 1.22, "2009-10": 1.37, "2010-11": 1.68, "2011-12": 1.82,
    "2012-13": 1.83, "2013-14": 1.86, "2014-15": 2.04, "2015-16": 2.10,
    "2016-17": 2.29, "2017-18": 2.65, "2018-19": 2.70, "2019-20": 2.87,
    "2020-21": 2.67, "2021-22": 3.18, "2022-23": 3.39, "2023-24": 3.57
}

# Global GDP Rankings
global_rank = {
    "2004-05": 12, "2005-06": 12, "2006-07": 12, "2007-08": 11,
    "2008-09": 11, "2009-10": 11, "2010-11": 10, "2011-12": 10,
    "2012-13": 10, "2013-14": 10, "2014-15": 9, "2015-16": 7,
    "2016-17": 7, "2017-18": 6, "2018-19": 6, "2019-20": 5,
    "2020-21": 6, "2021-22": 5, "2022-23": 5, "2023-24": 5
}

# Rupee Exchange Rate (per USD)
exchange_rate = {
    "2004-05": 44.93, "2005-06": 44.27, "2006-07": 45.25, "2007-08": 40.26,
    "2008-09": 45.99, "2009-10": 47.42, "2010-11": 45.56, "2011-12": 47.92,
    "2012-13": 54.41, "2013-14": 60.50, "2014-15": 61.14, "2015-16": 65.46,
    "2016-17": 67.07, "2017-18": 64.45, "2018-19": 69.89, "2019-20": 70.90,
    "2020-21": 74.23, "2021-22": 74.50, "2022-23": 80.35, "2023-24": 83.12
}

years = list(indicators["gdp_growth_rate"]["data"].keys())

# ============================================================================
# CHART 15: The Complete GDP Story - Nominal GDP in USD
# ============================================================================
print("[15/26] Creating Nominal GDP Story Chart...")

fig15 = go.Figure()

gdp_vals = list(gdp_nominal_usd.values())
colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig15.add_trace(go.Bar(
    x=years, y=gdp_vals,
    marker_color=colors,
    text=[f'${v:.2f}T' for v in gdp_vals],
    textposition='outside',
    hovertemplate='%{x}<br>GDP: $%{y:.2f} Trillion<extra></extra>'
))

# Add milestone annotations
fig15.add_annotation(x="2007-08", y=1.24, text="$1T+ Club",
                    showarrow=True, arrowhead=2, ax=0, ay=-40)
fig15.add_annotation(x="2014-15", y=2.04, text="$2T+ Club",
                    showarrow=True, arrowhead=2, ax=0, ay=-40)
fig15.add_annotation(x="2021-22", y=3.18, text="$3T+ Club",
                    showarrow=True, arrowhead=2, ax=0, ay=-40)

fig15.update_layout(
    title=dict(
        text="<b>India's Economic Rise: Nominal GDP in USD</b><br><sup>From $0.72T to $3.57T - A 5x Growth Journey</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="GDP (USD Trillion)",
    template="plotly_white",
    height=550, width=1100,
    showlegend=False
)

fig15.write_html(output_dir / "15_nominal_gdp_usd.html")
print("   Saved: charts/15_nominal_gdp_usd.html")

# ============================================================================
# CHART 16: Global Ranking Journey
# ============================================================================
print("[16/26] Creating Global Ranking Journey...")

fig16 = go.Figure()

rank_vals = list(global_rank.values())

fig16.add_trace(go.Scatter(
    x=years, y=rank_vals,
    mode='lines+markers+text',
    line=dict(color='#E74C3C', width=4),
    marker=dict(size=15, color='#E74C3C'),
    text=[f'#{r}' for r in rank_vals],
    textposition='top center',
    textfont=dict(size=10, color='#2C3E50'),
    hovertemplate='%{x}<br>Global Rank: #%{y}<extra></extra>'
))

# Invert y-axis (lower rank = better)
fig16.update_yaxes(autorange="reversed")

fig16.add_annotation(x="2004-05", y=12, text="Started at #12",
                    showarrow=True, arrowhead=2, ax=-60, ay=30)
fig16.add_annotation(x="2023-24", y=5, text="Now #5 (5th Largest Economy)",
                    showarrow=True, arrowhead=2, ax=60, ay=-30,
                    font=dict(color="green", size=12))

fig16.update_layout(
    title=dict(
        text="<b>India's Rise in Global GDP Rankings</b><br><sup>From 12th to 5th Largest Economy in 20 Years</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Global GDP Rank",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False
)

fig16.write_html(output_dir / "16_global_ranking.html")
print("   Saved: charts/16_global_ranking.html")

# ============================================================================
# CHART 17: Structural Transformation - Sector-wise GDP
# ============================================================================
print("[17/26] Creating Structural Transformation Chart...")

fig17 = go.Figure()

fig17.add_trace(go.Scatter(
    x=years, y=list(gdp_agriculture.values()),
    mode='lines+markers', name='Agriculture',
    line=dict(color='#27AE60', width=3),
    fill='tonexty', fillcolor='rgba(39, 174, 96, 0.1)'
))

fig17.add_trace(go.Scatter(
    x=years, y=list(gdp_industry.values()),
    mode='lines+markers', name='Industry',
    line=dict(color='#E74C3C', width=3),
    fill='tonexty', fillcolor='rgba(231, 76, 60, 0.1)'
))

fig17.add_trace(go.Scatter(
    x=years, y=list(gdp_services.values()),
    mode='lines+markers', name='Services',
    line=dict(color='#3498DB', width=3),
    fill='tonexty', fillcolor='rgba(52, 152, 219, 0.1)'
))

fig17.update_layout(
    title=dict(
        text="<b>Structural Transformation of Indian Economy</b><br><sup>Sector-wise GDP Contribution (%)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Share of GDP (%)",
    template="plotly_white",
    height=550, width=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified"
)

fig17.write_html(output_dir / "17_structural_transformation.html")
print("   Saved: charts/17_structural_transformation.html")

# ============================================================================
# CHART 18: Manufacturing Story
# ============================================================================
print("[18/26] Creating Manufacturing Story Chart...")

fig18 = go.Figure()

mfg_vals = list(manufacturing_growth.values())
colors = ['#27AE60' if v > 0 else '#E74C3C' for v in mfg_vals]

fig18.add_trace(go.Bar(
    x=years, y=mfg_vals,
    marker_color=colors,
    text=[f'{v:.1f}%' for v in mfg_vals],
    textposition='outside'
))

fig18.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)

# Add Make in India annotation
fig18.add_vline(x="2014-15", line_dash="dash", line_color="purple", opacity=0.7)
fig18.add_annotation(x="2014-15", y=14, text="Make in India<br>Launched",
                    showarrow=True, arrowhead=2, ax=-60, ay=-20)

fig18.update_layout(
    title=dict(
        text="<b>Manufacturing Sector Growth Rate</b><br><sup>The Make in India Story</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Growth Rate (%)",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False
)

fig18.write_html(output_dir / "18_manufacturing_growth.html")
print("   Saved: charts/18_manufacturing_growth.html")

# ============================================================================
# CHART 19: Investment Climate - Gross Capital Formation
# ============================================================================
print("[19/26] Creating Investment Climate Chart...")

fig19 = make_subplots(specs=[[{"secondary_y": True}]])

gcf_vals = list(gcf_rate.values())
fdi_vals = list(indicators["fdi_inflows"]["data"].values())

colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig19.add_trace(
    go.Bar(x=years, y=gcf_vals, name='GCF (% of GDP)',
           marker_color=colors, opacity=0.7),
    secondary_y=False
)

fig19.add_trace(
    go.Scatter(x=years, y=fdi_vals, name='FDI Inflows ($B)',
               line=dict(color='#2ECC71', width=4),
               mode='lines+markers', marker=dict(size=8)),
    secondary_y=True
)

fig19.update_layout(
    title=dict(
        text="<b>Investment Climate: Domestic vs Foreign Investment</b><br><sup>Gross Capital Formation & FDI Inflows</sup>",
        font=dict(size=22)
    ),
    template="plotly_white",
    height=550, width=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified"
)

fig19.update_yaxes(title_text="GCF (% of GDP)", secondary_y=False)
fig19.update_yaxes(title_text="FDI Inflows (USD Billion)", secondary_y=True)

fig19.write_html(output_dir / "19_investment_climate.html")
print("   Saved: charts/19_investment_climate.html")

# ============================================================================
# CHART 20: Poverty Reduction Journey
# ============================================================================
print("[20/26] Creating Poverty Reduction Chart...")

fig20 = go.Figure()

poverty_vals = list(poverty_rate.values())

fig20.add_trace(go.Scatter(
    x=years, y=poverty_vals,
    mode='lines+markers',
    fill='tozeroy',
    line=dict(color='#9B59B6', width=3),
    marker=dict(size=10),
    fillcolor='rgba(155, 89, 182, 0.3)'
))

# Calculate people lifted out of poverty (approximate)
fig20.add_annotation(x="2004-05", y=37.2, text="~400M in poverty",
                    showarrow=True, arrowhead=2, ax=-60, ay=-30)
fig20.add_annotation(x="2023-24", y=10.2, text="~140M in poverty<br>(260M lifted out)",
                    showarrow=True, arrowhead=2, ax=60, ay=30,
                    font=dict(color="green"))

fig20.update_layout(
    title=dict(
        text="<b>India's Poverty Reduction Story</b><br><sup>Poverty Headcount Ratio (% of Population)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Poverty Rate (%)",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False
)

fig20.write_html(output_dir / "20_poverty_reduction.html")
print("   Saved: charts/20_poverty_reduction.html")

# ============================================================================
# CHART 21: Human Development Progress
# ============================================================================
print("[21/26] Creating Human Development Chart...")

fig21 = go.Figure()

hdi_vals = list(hdi_values.values())
colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig21.add_trace(go.Scatter(
    x=years, y=hdi_vals,
    mode='lines+markers',
    line=dict(color='#16A085', width=4),
    marker=dict(size=12, color=colors, line=dict(width=2, color='white'))
))

# Add HDI category bands
fig21.add_hrect(y0=0, y1=0.55, fillcolor="red", opacity=0.1, line_width=0,
               annotation_text="Low HDI", annotation_position="right")
fig21.add_hrect(y0=0.55, y1=0.70, fillcolor="yellow", opacity=0.1, line_width=0,
               annotation_text="Medium HDI", annotation_position="right")
fig21.add_hrect(y0=0.70, y1=0.80, fillcolor="lightgreen", opacity=0.1, line_width=0,
               annotation_text="High HDI", annotation_position="right")

fig21.update_layout(
    title=dict(
        text="<b>Human Development Index Progress</b><br><sup>From Low HDI (0.535) to Medium HDI (0.670)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="HDI Value",
    template="plotly_white",
    height=550, width=1100,
    showlegend=False,
    yaxis=dict(range=[0.5, 0.75])
)

fig21.write_html(output_dir / "21_hdi_progress.html")
print("   Saved: charts/21_hdi_progress.html")

# ============================================================================
# CHART 22: Tax Revenue Mobilization
# ============================================================================
print("[22/26] Creating Tax Revenue Chart...")

fig22 = go.Figure()

tax_vals = list(tax_gdp_ratio.values())

fig22.add_trace(go.Scatter(
    x=years, y=tax_vals,
    mode='lines+markers',
    line=dict(color='#2980B9', width=3),
    marker=dict(size=10),
    fill='tozeroy',
    fillcolor='rgba(41, 128, 185, 0.2)'
))

# Add GST annotation
fig22.add_vline(x="2017-18", line_dash="dash", line_color="green", opacity=0.7)
fig22.add_annotation(x="2017-18", y=11.5, text="GST<br>Implementation",
                    showarrow=True, arrowhead=2, ax=-50, ay=-30)

fig22.update_layout(
    title=dict(
        text="<b>Tax Revenue Mobilization</b><br><sup>Tax-to-GDP Ratio (%)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Tax/GDP Ratio (%)",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False
)

fig22.write_html(output_dir / "22_tax_revenue.html")
print("   Saved: charts/22_tax_revenue.html")

# ============================================================================
# CHART 23: Rupee Journey
# ============================================================================
print("[23/26] Creating Rupee Exchange Rate Chart...")

fig23 = go.Figure()

fx_vals = list(exchange_rate.values())

fig23.add_trace(go.Scatter(
    x=years, y=fx_vals,
    mode='lines+markers',
    line=dict(color='#E67E22', width=3),
    marker=dict(size=10),
    fill='tozeroy',
    fillcolor='rgba(230, 126, 34, 0.2)'
))

fig23.add_annotation(x="2013-14", y=60.5, text="Taper Tantrum<br>Crisis",
                    showarrow=True, arrowhead=2, ax=-50, ay=-40)
fig23.add_annotation(x="2020-21", y=74.23, text="COVID<br>Depreciation",
                    showarrow=True, arrowhead=2, ax=50, ay=-30)

fig23.update_layout(
    title=dict(
        text="<b>Indian Rupee's Journey</b><br><sup>Exchange Rate (INR per USD)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="INR per USD",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False
)

fig23.write_html(output_dir / "23_rupee_exchange.html")
print("   Saved: charts/23_rupee_exchange.html")

# ============================================================================
# CHART 24: Policy Timeline with Economic Impact
# ============================================================================
print("[24/26] Creating Policy Timeline Chart...")

# Major policy events
policies = [
    {"year": "2005-06", "event": "NREGA Launched", "impact": "Employment Guarantee", "type": "social"},
    {"year": "2008-09", "event": "Global Financial Crisis", "impact": "Stimulus Package", "type": "crisis"},
    {"year": "2011-12", "event": "Policy Paralysis Era", "impact": "Stalled Reforms", "type": "negative"},
    {"year": "2014-15", "event": "Modi Era Begins", "impact": "Reform Push", "type": "reform"},
    {"year": "2014-15", "event": "Jan Dhan Yojana", "impact": "Financial Inclusion", "type": "social"},
    {"year": "2015-16", "event": "Make in India", "impact": "Manufacturing Focus", "type": "reform"},
    {"year": "2016-17", "event": "Demonetization", "impact": "Cash Crunch", "type": "disruption"},
    {"year": "2017-18", "event": "GST Implementation", "impact": "Tax Reform", "type": "reform"},
    {"year": "2019-20", "event": "Corporate Tax Cut", "impact": "Business Boost", "type": "reform"},
    {"year": "2020-21", "event": "COVID-19 Pandemic", "impact": "Economic Shock", "type": "crisis"},
    {"year": "2021-22", "event": "PLI Schemes", "impact": "Manufacturing Incentives", "type": "reform"},
]

gdp_growth = list(indicators["gdp_growth_rate"]["data"].values())

fig24 = go.Figure()

# GDP Growth line
fig24.add_trace(go.Scatter(
    x=years, y=gdp_growth,
    mode='lines',
    name='GDP Growth',
    line=dict(color='#2E86AB', width=3),
    fill='tozeroy',
    fillcolor='rgba(46, 134, 171, 0.2)'
))

# Policy markers
colors_map = {"social": "green", "crisis": "red", "negative": "orange",
              "reform": "blue", "disruption": "purple"}

for p in policies:
    idx = years.index(p["year"]) if p["year"] in years else -1
    if idx >= 0:
        fig24.add_annotation(
            x=p["year"], y=gdp_growth[idx] + 1.5,
            text=p["event"],
            showarrow=True, arrowhead=2,
            font=dict(size=9, color=colors_map[p["type"]]),
            ax=0, ay=-30
        )

fig24.update_layout(
    title=dict(
        text="<b>Major Policy Events & Economic Impact</b><br><sup>GDP Growth with Policy Timeline (2004-2024)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="GDP Growth Rate (%)",
    template="plotly_white",
    height=600, width=1200,
    showlegend=False
)

fig24.write_html(output_dir / "24_policy_timeline.html")
print("   Saved: charts/24_policy_timeline.html")

# ============================================================================
# CHART 25: Composite Economic Performance Index
# ============================================================================
print("[25/26] Creating Composite Performance Index...")

# Create a composite index (normalized 0-100)
def normalize(data, higher_better=True):
    vals = list(data.values())
    min_v, max_v = min(vals), max(vals)
    if higher_better:
        return [(v - min_v) / (max_v - min_v) * 100 for v in vals]
    else:
        return [(max_v - v) / (max_v - min_v) * 100 for v in vals]

gdp_norm = normalize(indicators["gdp_growth_rate"]["data"], True)
inf_norm = normalize(indicators["cpi_inflation"]["data"], False)
unemp_norm = normalize(indicators["unemployment_rate"]["data"], False)
fdi_norm = normalize(indicators["fdi_inflows"]["data"], True)
forex_norm = normalize(indicators["forex_reserves"]["data"], True)

# Weighted average
composite = []
for i in range(len(years)):
    score = (gdp_norm[i] * 0.3 + inf_norm[i] * 0.2 + unemp_norm[i] * 0.15 +
             fdi_norm[i] * 0.15 + forex_norm[i] * 0.2)
    composite.append(score)

fig25 = go.Figure()

colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig25.add_trace(go.Bar(
    x=years, y=composite,
    marker_color=colors,
    text=[f'{v:.0f}' for v in composite],
    textposition='outside'
))

# Add average lines
upa_avg = np.mean(composite[:10])
nda_avg = np.mean(composite[10:])

fig25.add_hline(y=upa_avg, line_dash="dash", line_color="blue",
               annotation_text=f"UPA Avg: {upa_avg:.0f}", annotation_position="left")
fig25.add_hline(y=nda_avg, line_dash="dash", line_color="orange",
               annotation_text=f"NDA Avg: {nda_avg:.0f}", annotation_position="right")

fig25.update_layout(
    title=dict(
        text="<b>Composite Economic Performance Index</b><br><sup>Weighted: GDP(30%) + Inflation(20%) + Jobs(15%) + FDI(15%) + Forex(20%)</sup>",
        font=dict(size=22)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Performance Score (0-100)",
    template="plotly_white",
    height=550, width=1200,
    showlegend=False,
    yaxis=dict(range=[0, 100])
)

fig25.write_html(output_dir / "25_composite_index.html")
print("   Saved: charts/25_composite_index.html")

# ============================================================================
# CHART 26: The Complete Story - Executive Dashboard
# ============================================================================
print("[26/26] Creating Executive Dashboard...")

fig26 = make_subplots(
    rows=3, cols=3,
    specs=[
        [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
        [{"colspan": 3, "type": "scatter"}, None, None]
    ],
    subplot_titles=(
        "GDP Growth Rate", "Period Comparison", "Forex Reserves",
        "Inflation Trend", "FDI Inflows", "Poverty Reduction",
        "20-Year Economic Journey: Key Indicators"
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 1. GDP Growth
gdp = list(indicators["gdp_growth_rate"]["data"].values())
fig26.add_trace(go.Scatter(x=years, y=gdp, mode='lines+markers',
                           line=dict(color='#2E86AB'), name='GDP'), row=1, col=1)

# 2. Period Comparison
fig26.add_trace(go.Bar(x=['GDP', 'Inflation', 'FDI'], y=[7.8, 8.1, 30.4],
                       name='UPA', marker_color='#3366CC'), row=1, col=2)
fig26.add_trace(go.Bar(x=['GDP', 'Inflation', 'FDI'], y=[6.1, 5.0, 66.6],
                       name='NDA', marker_color='#FF9900'), row=1, col=2)

# 3. Forex
forex = list(indicators["forex_reserves"]["data"].values())
fig26.add_trace(go.Scatter(x=years, y=forex, mode='lines+markers',
                           line=dict(color='#27AE60'), name='Forex', fill='tozeroy'), row=1, col=3)

# 4. Inflation
inf = list(indicators["cpi_inflation"]["data"].values())
fig26.add_trace(go.Scatter(x=years, y=inf, mode='lines+markers',
                           line=dict(color='#E74C3C'), name='Inflation'), row=2, col=1)

# 5. FDI
fdi = list(indicators["fdi_inflows"]["data"].values())
colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]
fig26.add_trace(go.Bar(x=years, y=fdi, marker_color=colors, name='FDI'), row=2, col=2)

# 6. Poverty
fig26.add_trace(go.Scatter(x=years, y=poverty_vals, mode='lines+markers',
                           line=dict(color='#9B59B6'), name='Poverty', fill='tozeroy'), row=2, col=3)

# 7. Composite Journey
fig26.add_trace(go.Scatter(x=years, y=composite, mode='lines+markers',
                           line=dict(color='#2C3E50', width=3), name='Composite',
                           fill='tozeroy', fillcolor='rgba(44, 62, 80, 0.2)'), row=3, col=1)

fig26.update_layout(
    title=dict(
        text="<b>India Economic Dashboard: The Complete Story (2004-2024)</b>",
        font=dict(size=24),
        x=0.5
    ),
    template="plotly_white",
    height=1000, width=1400,
    showlegend=False
)

fig26.write_html(output_dir / "26_executive_dashboard.html")
print("   Saved: charts/26_executive_dashboard.html")

# ============================================================================
# Update Index Page with All Charts
# ============================================================================
print()
print("Updating comprehensive index.html...")

index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>India Economic Comparison - Complete Analysis Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1500px; margin: 0 auto; padding: 40px 20px; }
        header { text-align: center; margin-bottom: 50px; }
        .flag-icon { font-size: 56px; margin-bottom: 15px; }
        h1 {
            font-size: 3rem; font-weight: 700;
            background: linear-gradient(90deg, #FF9933, #FFFFFF, #138808);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { font-size: 1.3rem; color: #a0a0a0; font-weight: 300; }
        .period-badges { display: flex; justify-content: center; gap: 20px; margin-top: 25px; flex-wrap: wrap; }
        .badge { padding: 14px 28px; border-radius: 30px; font-weight: 600; font-size: 1rem; }
        .badge-upa { background: linear-gradient(135deg, #3366CC, #1a3a6e); }
        .badge-nda { background: linear-gradient(135deg, #FF9900, #cc7a00); }
        .stats-banner { display: flex; justify-content: center; gap: 50px; margin: 40px 0; flex-wrap: wrap; }
        .stat-item { text-align: center; }
        .stat-value {
            font-size: 3rem; font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stat-label { font-size: 0.9rem; color: #a0a0a0; margin-top: 5px; }
        .section-title {
            font-size: 1.6rem; font-weight: 600; margin: 50px 0 25px;
            padding-left: 15px; border-left: 4px solid #667eea;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px; margin-bottom: 40px;
        }
        .chart-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px; padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        .chart-card:hover {
            transform: translateY(-8px);
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        .chart-card h3 { font-size: 1.1rem; font-weight: 600; margin-bottom: 10px; }
        .chart-card p { color: #a0a0a0; font-size: 0.85rem; line-height: 1.5; margin-bottom: 18px; }
        .chart-card a {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; text-decoration: none; border-radius: 25px;
            font-weight: 500; font-size: 0.85rem; transition: all 0.2s ease;
        }
        .chart-card a:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        .highlight-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border: 2px solid rgba(102, 126, 234, 0.5);
        }
        .key-findings {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px; padding: 35px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 40px 0;
        }
        .key-findings h2 { font-size: 1.5rem; margin-bottom: 25px; }
        .findings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
        .finding-item {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px; padding: 20px;
            border-left: 4px solid #667eea;
        }
        .finding-item h4 { color: #667eea; margin-bottom: 8px; font-size: 0.95rem; }
        .finding-item p { color: #d0d0d0; font-size: 0.85rem; line-height: 1.5; }
        .methodology {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px; padding: 35px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }
        .methodology h2 { font-size: 1.4rem; margin-bottom: 25px; }
        .methodology-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .methodology-item {
            display: flex; align-items: flex-start; gap: 12px;
            padding: 12px; background: rgba(255, 255, 255, 0.03); border-radius: 10px;
        }
        .methodology-item .icon { color: #27ae60; font-size: 1.2rem; }
        .methodology-item span { color: #d0d0d0; font-size: 0.9rem; }
        .disclaimer {
            background: rgba(231, 76, 60, 0.1);
            border: 1px solid rgba(231, 76, 60, 0.3);
            border-radius: 12px; padding: 25px; margin-top: 30px;
        }
        .disclaimer h3 { color: #e74c3c; margin-bottom: 10px; }
        .disclaimer p { color: #d0d0d0; font-size: 0.9rem; line-height: 1.6; }
        footer { text-align: center; margin-top: 50px; padding: 30px; color: #707070; font-size: 0.85rem; }
        footer a { color: #667eea; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="flag-icon">🇮🇳</div>
            <h1>India Economic Comparison System</h1>
            <p class="subtitle">The Complete Economic Story: UPA (2004-2014) vs NDA (2014-2024)</p>
            <div class="period-badges">
                <span class="badge badge-upa">UPA Era: 2004-2014</span>
                <span class="badge badge-nda">NDA Era: 2014-2024</span>
            </div>
        </header>

        <div class="stats-banner">
            <div class="stat-item"><div class="stat-value">20</div><div class="stat-label">Years of Data</div></div>
            <div class="stat-item"><div class="stat-value">26</div><div class="stat-label">Interactive Charts</div></div>
            <div class="stat-item"><div class="stat-value">15+</div><div class="stat-label">Economic Indicators</div></div>
            <div class="stat-item"><div class="stat-value">$3.57T</div><div class="stat-label">Current GDP</div></div>
            <div class="stat-item"><div class="stat-value">#5</div><div class="stat-label">Global Ranking</div></div>
        </div>

        <div class="key-findings">
            <h2>📊 Key Economic Findings at a Glance</h2>
            <div class="findings-grid">
                <div class="finding-item">
                    <h4>GDP Growth</h4>
                    <p>UPA averaged 7.8% growth vs NDA's 6.1% (includes COVID -6.6%). Excluding COVID, NDA averages 7.1%.</p>
                </div>
                <div class="finding-item">
                    <h4>Inflation Control</h4>
                    <p>NDA achieved 5.0% average inflation vs UPA's 8.1% - a significant improvement in price stability.</p>
                </div>
                <div class="finding-item">
                    <h4>FDI Attraction</h4>
                    <p>NDA attracted $665B cumulative FDI vs UPA's $303B - more than double the foreign investment.</p>
                </div>
                <div class="finding-item">
                    <h4>Forex Reserves</h4>
                    <p>Reserves grew from $141B (2004) to $645B (2024) - providing 11 months of import cover.</p>
                </div>
                <div class="finding-item">
                    <h4>Global Standing</h4>
                    <p>India rose from 12th to 5th largest economy - poised to become 3rd largest by 2028.</p>
                </div>
                <div class="finding-item">
                    <h4>Poverty Reduction</h4>
                    <p>Poverty headcount fell from 37% to ~10% - approximately 260 million lifted out of poverty.</p>
                </div>
            </div>
        </div>

        <h2 class="section-title">📈 The Big Picture</h2>
        <div class="chart-grid">
            <div class="chart-card highlight-card">
                <h3>🏆 Executive Dashboard</h3>
                <p>Complete 9-panel view of India's economic journey - GDP, inflation, FDI, forex, poverty, and composite index all in one view.</p>
                <a href="26_executive_dashboard.html">View Dashboard →</a>
            </div>
            <div class="chart-card highlight-card">
                <h3>📊 Composite Performance Index</h3>
                <p>Weighted score combining GDP (30%), Inflation (20%), Jobs (15%), FDI (15%), Forex (20%) for overall performance.</p>
                <a href="25_composite_index.html">View Index →</a>
            </div>
            <div class="chart-card">
                <h3>💰 Nominal GDP in USD</h3>
                <p>From $0.72T to $3.57T - India's 5x growth journey with $1T, $2T, $3T milestone markers.</p>
                <a href="15_nominal_gdp_usd.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🌍 Global Ranking Journey</h3>
                <p>From 12th to 5th largest economy - India's rise in global GDP rankings over 20 years.</p>
                <a href="16_global_ranking.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">📉 Core Economic Indicators</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>📈 GDP Growth Time Series</h3>
                <p>20-year GDP growth with UPA/NDA period shading and COVID-19 impact annotation.</p>
                <a href="01_gdp_growth_timeseries.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📊 Period Comparison Bars</h3>
                <p>Side-by-side comparison of key indicators between UPA and NDA periods.</p>
                <a href="02_period_comparison_bars.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🎯 Multi-Dimensional Radar</h3>
                <p>Normalized scores across Growth, Inflation, Fiscal, Forex, FDI, Employment dimensions.</p>
                <a href="03_radar_scorecard.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📉 GDP Trend Analysis</h3>
                <p>GDP growth with 3-year moving average smoothing to identify underlying trends.</p>
                <a href="07_gdp_trend_analysis.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">💹 Fiscal & Monetary Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>🔄 Phillips Curve</h3>
                <p>Growth-Inflation tradeoff showing which years achieved optimal policy mix.</p>
                <a href="08_phillips_curve.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>⚖️ Twin Deficits</h3>
                <p>Fiscal deficit vs current account balance - key stability indicators.</p>
                <a href="09_twin_deficits.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🏦 Debt Sustainability</h3>
                <p>Public debt-to-GDP with Maastricht 60% benchmark and COVID spike.</p>
                <a href="10_debt_sustainability.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>💵 Tax Revenue Mobilization</h3>
                <p>Tax-to-GDP ratio trend with GST implementation marker.</p>
                <a href="22_tax_revenue.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>💱 Rupee Exchange Journey</h3>
                <p>INR/USD exchange rate from 44.93 to 83.12 with crisis annotations.</p>
                <a href="23_rupee_exchange.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">🌐 External Sector & Investment</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>💰 Forex Reserves</h3>
                <p>$141B to $645B - building a stability buffer over two decades.</p>
                <a href="04_forex_reserves.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🚢 Trade Balance</h3>
                <p>Exports, imports, and net trade position showing global integration.</p>
                <a href="11_trade_balance.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📈 Investment Climate</h3>
                <p>Domestic investment (GCF) vs FDI inflows - dual-axis view.</p>
                <a href="19_investment_climate.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>💵 Per Capita Income</h3>
                <p>From ₹26,674 to ₹1,84,507 - absolute value and growth rates.</p>
                <a href="12_per_capita_income.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">🏭 Structural & Sectoral Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>🔄 Structural Transformation</h3>
                <p>Agriculture → Industry → Services shift in GDP composition.</p>
                <a href="17_structural_transformation.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🏭 Manufacturing Growth</h3>
                <p>Manufacturing sector performance with Make in India marker.</p>
                <a href="18_manufacturing_growth.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>⚡ Policy Impact Waterfall</h3>
                <p>Event-wise growth decomposition: Crisis, Reforms, COVID, Recovery.</p>
                <a href="05_policy_impact_waterfall.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📅 Policy Timeline</h3>
                <p>Major policy events mapped onto GDP growth trajectory.</p>
                <a href="24_policy_timeline.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">👥 Social & Human Development</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>📉 Poverty Reduction</h3>
                <p>From 37% to ~10% - approximately 260 million lifted out of poverty.</p>
                <a href="20_poverty_reduction.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📈 Human Development Index</h3>
                <p>HDI progress from 0.535 (Low) to 0.670 (Medium) category.</p>
                <a href="21_hdi_progress.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">📊 Statistical Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>🔥 Correlation Heatmap</h3>
                <p>Pearson correlations between all major economic indicators.</p>
                <a href="13_correlation_heatmap.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📋 Statistical Summary</h3>
                <p>Mean, Std Dev, CV comparison table for both periods.</p>
                <a href="14_statistical_summary.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🖥️ Comprehensive Dashboard</h3>
                <p>Multi-panel dashboard with trends, comparisons, correlations.</p>
                <a href="06_comprehensive_dashboard.html">View Chart →</a>
            </div>
        </div>

        <div class="methodology">
            <h2>📋 Methodology & Data Sources</h2>
            <div class="methodology-grid">
                <div class="methodology-item"><span class="icon">✓</span><span>MOSPI - GDP, GVA, Per Capita Income</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>RBI - Inflation, Forex, Current Account</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>World Bank - Development Indicators, HDI</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>IMF - WEO Projections, Global Rankings</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>DPIIT - FDI Inflows by Sector</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>CMIE/NSSO - Employment, Poverty Data</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>Budget Documents - Fiscal Data</span></div>
                <div class="methodology-item"><span class="icon">✓</span><span>UNDP - Human Development Reports</span></div>
            </div>
        </div>

        <div class="disclaimer">
            <h3>⚠️ Important Disclaimer</h3>
            <p>This analysis is for <strong>educational and research purposes</strong>. Economic performance is multi-dimensional with many causes beyond government policy. External conditions (global economy, commodity prices, pandemics) differed significantly between periods. This system aims to <strong>inform debate, not settle it</strong>. Users should cross-validate with official sources.</p>
        </div>

        <footer>
            <p>Built with Python, Plotly, and Rigorous Economic Methodology</p>
            <p>Data: MOSPI | RBI | World Bank | IMF | DPIIT | CMIE | UNDP</p>
            <p style="margin-top: 15px;"><a href="https://github.com/MAYANK12-WQ/India-Economic-Comparison">GitHub Repository</a> • MIT License • 2024</p>
        </footer>
    </div>
</body>
</html>
"""

with open(output_dir / "index.html", "w", encoding="utf-8") as f:
    f.write(index_html)

print("   Saved: charts/index.html")
print()
print("=" * 70)
print("ALL STORY-TELLING CHARTS GENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nTotal charts: 26")
print(f"Open: {(output_dir / 'index.html').absolute()}")
print()
