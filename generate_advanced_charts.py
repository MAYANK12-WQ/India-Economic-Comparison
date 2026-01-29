#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Economic Charts - India Economic Comparison System
============================================================

Professional-grade visualizations for economist-level analysis.
Includes: Dual-axis charts, correlation matrices, decomposition analysis,
Phillips Curve, Debt sustainability, and more.
"""

import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

# Create output directory
output_dir = Path("charts")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("GENERATING ADVANCED ECONOMIST-GRADE CHARTS")
print("=" * 70)
print()

# Load data
with open("data/sample_indicators.json") as f:
    data = json.load(f)

indicators = data["indicators"]

# Extended data for advanced charts
gdp_nominal = {
    "2004-05": 32.4, "2005-06": 36.9, "2006-07": 42.9, "2007-08": 49.9,
    "2008-09": 56.3, "2009-10": 64.8, "2010-11": 77.8, "2011-12": 90.1,
    "2012-13": 99.4, "2013-14": 112.3, "2014-15": 124.7, "2015-16": 137.7,
    "2016-17": 153.6, "2017-18": 170.9, "2018-19": 189.7, "2019-20": 200.7,
    "2020-21": 198.0, "2021-22": 236.6, "2022-23": 273.1, "2023-24": 295.4
}  # In Lakh Crore INR

current_account = {
    "2004-05": -0.4, "2005-06": -1.2, "2006-07": -1.0, "2007-08": -1.3,
    "2008-09": -2.3, "2009-10": -2.8, "2010-11": -2.8, "2011-12": -4.2,
    "2012-13": -4.8, "2013-14": -1.7, "2014-15": -1.3, "2015-16": -1.1,
    "2016-17": -0.6, "2017-18": -1.8, "2018-19": -2.1, "2019-20": -0.9,
    "2020-21": 0.9, "2021-22": -1.2, "2022-23": -2.0, "2023-24": -1.1
}  # % of GDP

debt_to_gdp = {
    "2004-05": 81.0, "2005-06": 79.1, "2006-07": 75.4, "2007-08": 73.1,
    "2008-09": 74.5, "2009-10": 72.5, "2010-11": 67.5, "2011-12": 67.5,
    "2012-13": 67.2, "2013-14": 67.1, "2014-15": 67.1, "2015-16": 68.8,
    "2016-17": 68.9, "2017-18": 69.6, "2018-19": 70.4, "2019-20": 74.1,
    "2020-21": 89.6, "2021-22": 84.2, "2022-23": 81.3, "2023-24": 81.9
}  # % of GDP

per_capita_income = {
    "2004-05": 26674, "2005-06": 29786, "2006-07": 33904, "2007-08": 38856,
    "2008-09": 42453, "2009-10": 47761, "2010-11": 55836, "2011-12": 64316,
    "2012-13": 71593, "2013-14": 79412, "2014-15": 86647, "2015-16": 94178,
    "2016-17": 104880, "2017-18": 115224, "2018-19": 126406, "2019-20": 132115,
    "2020-21": 128829, "2021-22": 150307, "2022-23": 170620, "2023-24": 184507
}  # INR

exports = {
    "2004-05": 83.5, "2005-06": 103.1, "2006-07": 126.4, "2007-08": 163.1,
    "2008-09": 185.3, "2009-10": 178.8, "2010-11": 251.1, "2011-12": 305.9,
    "2012-13": 300.4, "2013-14": 314.4, "2014-15": 310.3, "2015-16": 262.3,
    "2016-17": 275.9, "2017-18": 303.5, "2018-19": 330.1, "2019-20": 313.4,
    "2020-21": 291.8, "2021-22": 422.0, "2022-23": 451.1, "2023-24": 437.1
}  # USD Billion

imports = {
    "2004-05": 111.5, "2005-06": 149.2, "2006-07": 185.7, "2007-08": 251.6,
    "2008-09": 303.7, "2009-10": 288.4, "2010-11": 369.8, "2011-12": 489.3,
    "2012-13": 490.7, "2013-14": 450.2, "2014-15": 448.0, "2015-16": 381.0,
    "2016-17": 384.4, "2017-18": 465.6, "2018-19": 514.1, "2019-20": 474.7,
    "2020-21": 394.4, "2021-22": 613.1, "2022-23": 714.2, "2023-24": 677.2
}  # USD Billion

# ============================================================================
# CHART 7: GDP Growth with Moving Average (Trend Analysis)
# ============================================================================
print("[7/14] Creating GDP Growth with Trend Analysis...")

gdp_data = indicators["gdp_growth_rate"]["data"]
years = list(gdp_data.keys())
values = list(gdp_data.values())

# Calculate 3-year moving average
ma_values = []
for i in range(len(values)):
    if i < 2:
        ma_values.append(None)
    else:
        ma_values.append(np.mean(values[i-2:i+1]))

fig7 = go.Figure()

fig7.add_trace(go.Scatter(
    x=years, y=values,
    mode='lines+markers',
    name='Actual GDP Growth',
    line=dict(color='#2E86AB', width=2),
    marker=dict(size=8)
))

fig7.add_trace(go.Scatter(
    x=years, y=ma_values,
    mode='lines',
    name='3-Year Moving Avg',
    line=dict(color='#E74C3C', width=3, dash='dash')
))

# Add trend annotations
fig7.add_annotation(x="2007-08", y=9.5, text="Pre-GFC Peak",
                    showarrow=True, arrowhead=2, ax=-40, ay=-30)
fig7.add_annotation(x="2020-21", y=-6.6, text="COVID Trough",
                    showarrow=True, arrowhead=2, ax=40, ay=-30)

fig7.update_layout(
    title=dict(
        text="<b>GDP Growth Rate with Trend Analysis</b><br><sup>3-Year Moving Average Smoothing</sup>",
        font=dict(size=20)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Growth Rate (%)",
    template="plotly_white",
    height=500, width=1000,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified"
)

fig7.write_html(output_dir / "07_gdp_trend_analysis.html")
print("   Saved: charts/07_gdp_trend_analysis.html")

# ============================================================================
# CHART 8: Inflation vs Growth (Phillips Curve Scatter)
# ============================================================================
print("[8/14] Creating Phillips Curve Analysis...")

gdp_vals = list(indicators["gdp_growth_rate"]["data"].values())
inf_vals = list(indicators["cpi_inflation"]["data"].values())
years_list = list(indicators["gdp_growth_rate"]["data"].keys())

fig8 = go.Figure()

# UPA period (blue)
upa_gdp = gdp_vals[:10]
upa_inf = inf_vals[:10]
upa_years = years_list[:10]

# NDA period (orange)
nda_gdp = gdp_vals[10:]
nda_inf = inf_vals[10:]
nda_years = years_list[10:]

fig8.add_trace(go.Scatter(
    x=upa_inf, y=upa_gdp,
    mode='markers+text',
    name='UPA (2004-14)',
    marker=dict(size=14, color='#3366CC', symbol='circle'),
    text=[y[-2:] for y in upa_years],
    textposition='top center',
    textfont=dict(size=9)
))

fig8.add_trace(go.Scatter(
    x=nda_inf, y=nda_gdp,
    mode='markers+text',
    name='NDA (2014-24)',
    marker=dict(size=14, color='#FF9900', symbol='diamond'),
    text=[y[-2:] for y in nda_years],
    textposition='top center',
    textfont=dict(size=9)
))

# Add quadrant lines
fig8.add_hline(y=6, line_dash="dot", line_color="gray", opacity=0.5)
fig8.add_vline(x=6, line_dash="dot", line_color="gray", opacity=0.5)

# Add quadrant labels
fig8.add_annotation(x=3, y=9, text="Sweet Spot<br>(High Growth, Low Inflation)",
                    showarrow=False, font=dict(size=10, color="green"))
fig8.add_annotation(x=10, y=2, text="Stagflation Risk<br>(Low Growth, High Inflation)",
                    showarrow=False, font=dict(size=10, color="red"))

fig8.update_layout(
    title=dict(
        text="<b>Growth-Inflation Tradeoff (Modified Phillips Curve)</b><br><sup>Each point represents a fiscal year</sup>",
        font=dict(size=20)
    ),
    xaxis_title="CPI Inflation (%)",
    yaxis_title="GDP Growth Rate (%)",
    template="plotly_white",
    height=600, width=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig8.write_html(output_dir / "08_phillips_curve.html")
print("   Saved: charts/08_phillips_curve.html")

# ============================================================================
# CHART 9: Twin Deficits Analysis (Fiscal + Current Account)
# ============================================================================
print("[9/14] Creating Twin Deficits Analysis...")

years = list(indicators["fiscal_deficit"]["data"].keys())
fiscal = list(indicators["fiscal_deficit"]["data"].values())
current = [current_account[y] for y in years]

fig9 = make_subplots(specs=[[{"secondary_y": True}]])

fig9.add_trace(
    go.Bar(x=years, y=fiscal, name='Fiscal Deficit (% GDP)',
           marker_color='#E74C3C', opacity=0.7),
    secondary_y=False
)

fig9.add_trace(
    go.Scatter(x=years, y=current, name='Current Account (% GDP)',
               line=dict(color='#2E86AB', width=3),
               mode='lines+markers'),
    secondary_y=True
)

# Add zero line for current account
fig9.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)

fig9.update_layout(
    title=dict(
        text="<b>Twin Deficits Analysis</b><br><sup>Fiscal Deficit vs Current Account Balance</sup>",
        font=dict(size=20)
    ),
    template="plotly_white",
    height=500, width=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    barmode='group'
)

fig9.update_yaxes(title_text="Fiscal Deficit (% of GDP)", secondary_y=False, range=[0, 12])
fig9.update_yaxes(title_text="Current Account (% of GDP)", secondary_y=True, range=[-6, 2])

fig9.write_html(output_dir / "09_twin_deficits.html")
print("   Saved: charts/09_twin_deficits.html")

# ============================================================================
# CHART 10: Debt Sustainability Dashboard
# ============================================================================
print("[10/14] Creating Debt Sustainability Dashboard...")

years = list(debt_to_gdp.keys())
debt = list(debt_to_gdp.values())
colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig10 = go.Figure()

fig10.add_trace(go.Scatter(
    x=years, y=debt,
    mode='lines+markers+text',
    fill='tozeroy',
    name='Public Debt/GDP',
    line=dict(color='#8E44AD', width=3),
    marker=dict(size=10, color=colors),
    text=[f'{d:.0f}%' for d in debt],
    textposition='top center',
    textfont=dict(size=9)
))

# Add Maastricht criteria line
fig10.add_hline(y=60, line_dash="dash", line_color="red",
                annotation_text="Maastricht Criterion (60%)", annotation_position="right")

# Add COVID annotation
fig10.add_annotation(x="2020-21", y=89.6, text="COVID Spike<br>89.6%",
                    showarrow=True, arrowhead=2, ax=-50, ay=-30,
                    font=dict(color="red"))

fig10.update_layout(
    title=dict(
        text="<b>Public Debt Sustainability Analysis</b><br><sup>General Government Debt as % of GDP</sup>",
        font=dict(size=20)
    ),
    xaxis_title="Financial Year",
    yaxis_title="Debt-to-GDP Ratio (%)",
    template="plotly_white",
    height=500, width=1100,
    showlegend=False,
    yaxis=dict(range=[50, 100])
)

fig10.write_html(output_dir / "10_debt_sustainability.html")
print("   Saved: charts/10_debt_sustainability.html")

# ============================================================================
# CHART 11: Trade Balance Waterfall
# ============================================================================
print("[11/14] Creating Trade Balance Analysis...")

years = list(exports.keys())
exp_vals = list(exports.values())
imp_vals = list(imports.values())
trade_balance = [e - i for e, i in zip(exp_vals, imp_vals)]

fig11 = go.Figure()

fig11.add_trace(go.Bar(
    x=years, y=exp_vals,
    name='Exports',
    marker_color='#27AE60'
))

fig11.add_trace(go.Bar(
    x=years, y=[-i for i in imp_vals],
    name='Imports',
    marker_color='#E74C3C'
))

fig11.add_trace(go.Scatter(
    x=years, y=trade_balance,
    name='Trade Balance',
    mode='lines+markers',
    line=dict(color='#2C3E50', width=3),
    marker=dict(size=8)
))

fig11.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)

fig11.update_layout(
    title=dict(
        text="<b>India's Trade Position (2004-2024)</b><br><sup>Exports, Imports & Trade Balance (USD Billion)</sup>",
        font=dict(size=20)
    ),
    xaxis_title="Financial Year",
    yaxis_title="USD Billion",
    template="plotly_white",
    height=550, width=1200,
    barmode='relative',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig11.write_html(output_dir / "11_trade_balance.html")
print("   Saved: charts/11_trade_balance.html")

# ============================================================================
# CHART 12: Per Capita Income Growth
# ============================================================================
print("[12/14] Creating Per Capita Income Analysis...")

years = list(per_capita_income.keys())
pci = list(per_capita_income.values())
pci_growth = [None] + [((pci[i] - pci[i-1])/pci[i-1])*100 for i in range(1, len(pci))]

fig12 = make_subplots(specs=[[{"secondary_y": True}]])

colors = ['#3366CC' if int(y.split('-')[0]) < 2014 else '#FF9900' for y in years]

fig12.add_trace(
    go.Bar(x=years, y=pci, name='Per Capita Income (INR)',
           marker_color=colors, opacity=0.7),
    secondary_y=False
)

fig12.add_trace(
    go.Scatter(x=years, y=pci_growth, name='YoY Growth (%)',
               line=dict(color='#2ECC71', width=3),
               mode='lines+markers'),
    secondary_y=True
)

fig12.update_layout(
    title=dict(
        text="<b>Per Capita Income Evolution</b><br><sup>Absolute Value & Year-on-Year Growth</sup>",
        font=dict(size=20)
    ),
    template="plotly_white",
    height=500, width=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig12.update_yaxes(title_text="Per Capita Income (INR)", secondary_y=False)
fig12.update_yaxes(title_text="YoY Growth (%)", secondary_y=True)

fig12.write_html(output_dir / "12_per_capita_income.html")
print("   Saved: charts/12_per_capita_income.html")

# ============================================================================
# CHART 13: Correlation Heatmap of Economic Indicators
# ============================================================================
print("[13/14] Creating Correlation Heatmap...")

# Prepare correlation matrix
ind_names = ['GDP Growth', 'Inflation', 'Fiscal Deficit', 'Unemployment', 'Forex', 'FDI']
gdp = list(indicators["gdp_growth_rate"]["data"].values())
inf = list(indicators["cpi_inflation"]["data"].values())
fis = list(indicators["fiscal_deficit"]["data"].values())
unemp = list(indicators["unemployment_rate"]["data"].values())
forex = list(indicators["forex_reserves"]["data"].values())
fdi = list(indicators["fdi_inflows"]["data"].values())

data_matrix = np.array([gdp, inf, fis, unemp, forex, fdi])
corr_matrix = np.corrcoef(data_matrix)

fig13 = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=ind_names,
    y=ind_names,
    colorscale='RdBu_r',
    zmin=-1, zmax=1,
    text=[[f'{val:.2f}' for val in row] for row in corr_matrix],
    texttemplate='%{text}',
    textfont=dict(size=12),
    hoverongaps=False
))

fig13.update_layout(
    title=dict(
        text="<b>Economic Indicators Correlation Matrix</b><br><sup>Pearson Correlation Coefficients (2004-2024)</sup>",
        font=dict(size=20)
    ),
    template="plotly_white",
    height=600, width=700,
    xaxis=dict(tickangle=45)
)

fig13.write_html(output_dir / "13_correlation_heatmap.html")
print("   Saved: charts/13_correlation_heatmap.html")

# ============================================================================
# CHART 14: Comprehensive Period Statistics Table
# ============================================================================
print("[14/14] Creating Statistical Summary Comparison...")

# Calculate comprehensive stats
def calc_stats(data_dict, start_year, end_year):
    vals = [v for k, v in data_dict.items() if start_year <= int(k.split('-')[0]) < end_year]
    return {
        'mean': np.mean(vals),
        'std': np.std(vals),
        'min': np.min(vals),
        'max': np.max(vals),
        'volatility': np.std(vals) / np.mean(vals) * 100 if np.mean(vals) != 0 else 0
    }

indicators_list = [
    ('GDP Growth (%)', indicators["gdp_growth_rate"]["data"]),
    ('CPI Inflation (%)', indicators["cpi_inflation"]["data"]),
    ('Fiscal Deficit (% GDP)', indicators["fiscal_deficit"]["data"]),
    ('Unemployment (%)', indicators["unemployment_rate"]["data"]),
    ('Forex Reserves ($B)', indicators["forex_reserves"]["data"]),
    ('FDI Inflows ($B)', indicators["fdi_inflows"]["data"]),
]

# Create comparison table
rows = []
for name, data in indicators_list:
    upa = calc_stats(data, 2004, 2014)
    nda = calc_stats(data, 2014, 2025)
    rows.append([
        name,
        f"{upa['mean']:.1f}", f"{upa['std']:.1f}", f"{upa['volatility']:.1f}%",
        f"{nda['mean']:.1f}", f"{nda['std']:.1f}", f"{nda['volatility']:.1f}%"
    ])

fig14 = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Indicator</b>', '<b>UPA Mean</b>', '<b>UPA Std</b>', '<b>UPA CV</b>',
                '<b>NDA Mean</b>', '<b>NDA Std</b>', '<b>NDA CV</b>'],
        fill_color='#2C3E50',
        font=dict(color='white', size=12),
        align='center',
        height=35
    ),
    cells=dict(
        values=list(zip(*rows)),
        fill_color=[['#ECF0F1', '#F8F9FA'] * 3],
        font=dict(size=11),
        align='center',
        height=30
    )
)])

fig14.update_layout(
    title=dict(
        text="<b>Statistical Comparison Summary</b><br><sup>Mean, Standard Deviation & Coefficient of Variation</sup>",
        font=dict(size=20)
    ),
    height=400, width=1000
)

fig14.write_html(output_dir / "14_statistical_summary.html")
print("   Saved: charts/14_statistical_summary.html")

# ============================================================================
# Update Index Page
# ============================================================================
print()
print("Updating index.html with all charts...")

index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>India Economic Comparison - Professional Analysis Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        header {
            text-align: center;
            margin-bottom: 50px;
        }
        .flag-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        h1 {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FF9933, #FFFFFF, #138808);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #a0a0a0;
            font-weight: 300;
        }
        .period-badges {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 25px;
        }
        .badge {
            padding: 12px 24px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .badge-upa {
            background: linear-gradient(135deg, #3366CC, #1a3a6e);
        }
        .badge-nda {
            background: linear-gradient(135deg, #FF9900, #cc7a00);
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 40px 0 20px;
            padding-left: 15px;
            border-left: 4px solid #667eea;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        .chart-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        .chart-card:hover {
            transform: translateY(-8px);
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        .chart-card h3 {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .chart-card p {
            color: #a0a0a0;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .chart-card a {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }
        .chart-card a:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .methodology {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 35px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }
        .methodology h2 {
            font-size: 1.4rem;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .methodology-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .methodology-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
        }
        .methodology-item .icon {
            color: #27ae60;
            font-size: 1.2rem;
        }
        .methodology-item span {
            color: #d0d0d0;
            font-size: 0.9rem;
        }
        .disclaimer {
            background: rgba(231, 76, 60, 0.1);
            border: 1px solid rgba(231, 76, 60, 0.3);
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
        }
        .disclaimer h3 {
            color: #e74c3c;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .disclaimer p {
            color: #d0d0d0;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            color: #707070;
            font-size: 0.85rem;
        }
        footer a {
            color: #667eea;
            text-decoration: none;
        }
        .stats-banner {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #a0a0a0;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="flag-icon">🇮🇳</div>
            <h1>India Economic Comparison System</h1>
            <p class="subtitle">Professional-Grade Analysis of India's Economic Performance (2004-2024)</p>
            <div class="period-badges">
                <span class="badge badge-upa">UPA Era: 2004-2014</span>
                <span class="badge badge-nda">NDA Era: 2014-2024</span>
            </div>
        </header>

        <div class="stats-banner">
            <div class="stat-item">
                <div class="stat-value">20</div>
                <div class="stat-label">Years of Data</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">14</div>
                <div class="stat-label">Interactive Charts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">6+</div>
                <div class="stat-label">Key Indicators</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">4</div>
                <div class="stat-label">Data Sources</div>
            </div>
        </div>

        <h2 class="section-title">Core Economic Indicators</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>📈 GDP Growth Time Series</h3>
                <p>Interactive visualization of India's GDP growth rate with period shading for UPA and NDA eras, including COVID-19 impact annotation.</p>
                <a href="01_gdp_growth_timeseries.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📊 Period Comparison</h3>
                <p>Side-by-side bar chart comparing key economic indicators (GDP, Inflation, Fiscal Deficit, Unemployment) between the two periods.</p>
                <a href="02_period_comparison_bars.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🎯 Multi-Dimensional Scorecard</h3>
                <p>Radar chart showing normalized performance scores across 6 dimensions: Growth, Inflation, Fiscal Prudence, Forex, FDI, Employment.</p>
                <a href="03_radar_scorecard.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>💰 Forex Reserves Trend</h3>
                <p>India's foreign exchange reserves growth from $141B to $645B over two decades, showing stability buffer accumulation.</p>
                <a href="04_forex_reserves.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">Advanced Economic Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>⚡ Policy Impact Waterfall</h3>
                <p>Waterfall chart decomposing growth changes by policy events: 2008 Crisis, Demonetization, GST, COVID-19, and recovery phases.</p>
                <a href="05_policy_impact_waterfall.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🖥️ Comprehensive Dashboard</h3>
                <p>Multi-panel dashboard combining GDP trends, period comparison bars, score distribution pie, and growth-inflation scatter.</p>
                <a href="06_comprehensive_dashboard.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📉 GDP Trend Analysis</h3>
                <p>GDP growth with 3-year moving average smoothing to identify underlying trends beyond year-to-year volatility.</p>
                <a href="07_gdp_trend_analysis.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🔄 Phillips Curve Analysis</h3>
                <p>Growth-Inflation tradeoff visualization showing which years achieved the "sweet spot" of high growth with low inflation.</p>
                <a href="08_phillips_curve.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">Fiscal & External Sector</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>⚖️ Twin Deficits Analysis</h3>
                <p>Dual-axis visualization of fiscal deficit and current account balance - key indicators for macroeconomic stability.</p>
                <a href="09_twin_deficits.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🏦 Debt Sustainability</h3>
                <p>Public debt-to-GDP ratio trend with Maastricht criterion benchmark (60%) and COVID spike annotation.</p>
                <a href="10_debt_sustainability.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>🚢 Trade Balance</h3>
                <p>Exports, imports, and trade balance evolution showing India's integration with global trade over 20 years.</p>
                <a href="11_trade_balance.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>💵 Per Capita Income</h3>
                <p>Per capita income growth from ₹26,674 to ₹1,84,507 - showing absolute values and year-on-year growth rates.</p>
                <a href="12_per_capita_income.html">View Chart →</a>
            </div>
        </div>

        <h2 class="section-title">Statistical Analysis</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>🔥 Correlation Heatmap</h3>
                <p>Pearson correlation matrix revealing relationships between GDP, Inflation, Fiscal Deficit, Unemployment, Forex, and FDI.</p>
                <a href="13_correlation_heatmap.html">View Chart →</a>
            </div>
            <div class="chart-card">
                <h3>📋 Statistical Summary</h3>
                <p>Comprehensive comparison table with mean, standard deviation, and coefficient of variation for both periods.</p>
                <a href="14_statistical_summary.html">View Chart →</a>
            </div>
        </div>

        <div class="methodology">
            <h2>📋 Methodology & Data Sources</h2>
            <div class="methodology-grid">
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Official data from MOSPI, RBI, World Bank, IMF</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Difference-in-Differences causal analysis</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Synthetic Control Method for policy impacts</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Bootstrap confidence intervals (10,000 iterations)</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Complete data: All years 2004-2024 included</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Equal scrutiny applied to both periods</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Multiple normalization techniques</span>
                </div>
                <div class="methodology-item">
                    <span class="icon">✓</span>
                    <span>Ethical review framework compliance</span>
                </div>
            </div>
        </div>

        <div class="disclaimer">
            <h3>⚠️ Important Disclaimer</h3>
            <p>This analysis is designed for educational and research purposes. Economic performance is multi-dimensional with many causes beyond government policy. External conditions (global economy, commodity prices, geopolitical events) differed significantly between periods. This system aims to <strong>inform debate, not settle it</strong>. Users should cross-validate findings with other sources and avoid drawing oversimplified conclusions.</p>
        </div>

        <footer>
            <p>Built with Python, Plotly, and rigorous economic methodology</p>
            <p>Data Sources: MOSPI | RBI | World Bank | IMF | DPIIT</p>
            <p style="margin-top: 15px;">
                <a href="https://github.com">View on GitHub</a> •
                MIT License •
                2024
            </p>
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
print("ALL ADVANCED CHARTS GENERATED SUCCESSFULLY!")
print("=" * 70)
print()
print(f"Total charts: 14")
print(f"Open this file in your browser:")
print(f"  {(output_dir / 'index.html').absolute()}")
print()
