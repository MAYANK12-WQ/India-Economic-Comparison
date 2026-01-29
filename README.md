<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Plotly-5.0+-green.svg" alt="Plotly">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Analysis-2004--2024-orange.svg" alt="Period">
  <img src="https://img.shields.io/badge/Charts-14-purple.svg" alt="Charts">
</p>

<h1 align="center">India Economic Comparison System</h1>

<p align="center">
  <strong>A Comprehensive, Transparent, and Methodologically Rigorous Economic Analysis Platform</strong>
</p>

<p align="center">
  Comparing India's Economic Performance: UPA Era (2004-2014) vs NDA Era (2014-2024)
</p>

---

## Overview

This project provides **professional-grade economic analysis** comparing India's macroeconomic performance across two distinct political periods. Built with the rigor expected by economists and policy analysts, it features:

- **14 Interactive Visualizations** powered by Plotly
- **20 Years of Economic Data** (2004-2024)
- **6+ Key Macroeconomic Indicators**
- **Multiple Analytical Frameworks** (Causal Inference, Counterfactual Analysis, Uncertainty Quantification)
- **Ethical Review Framework** ensuring balanced, unbiased analysis

---

## Key Economic Indicators Analyzed

| Category | Indicators | Data Source |
|----------|-----------|-------------|
| **Growth** | GDP Growth Rate, Per Capita Income, Industrial Production | MOSPI |
| **Inflation** | CPI Combined, Food Inflation, WPI | RBI |
| **Fiscal** | Fiscal Deficit (% GDP), Public Debt, Tax/GDP Ratio | Budget Documents |
| **External** | Current Account, Forex Reserves, FDI Inflows, Trade Balance | RBI, DPIIT |
| **Employment** | Unemployment Rate, Labor Force Participation | CMIE, NSSO |

---

## Interactive Charts Gallery

### Core Economic Indicators

| Chart | Description | Key Insight |
|-------|-------------|-------------|
| **GDP Growth Time Series** | 20-year GDP growth with period shading | Pre-GFC peak (9.6%), COVID trough (-6.6%) |
| **Period Comparison Bars** | Side-by-side indicator comparison | UPA avg: 7.8% growth, NDA avg: 6.1% growth |
| **Multi-Dimensional Radar** | Normalized scores across 6 dimensions | Trade-offs between growth and stability |
| **Forex Reserves Trend** | $141B to $645B accumulation | 4.6x growth in reserves over 20 years |

### Advanced Economic Analysis

| Chart | Description | Economist Use Case |
|-------|-------------|-------------------|
| **Phillips Curve Analysis** | Growth-Inflation scatter plot | Identify optimal policy mix years |
| **Twin Deficits Analysis** | Fiscal + Current Account dual-axis | Assess macroeconomic stability |
| **Debt Sustainability** | Debt/GDP with Maastricht benchmark | Evaluate fiscal headroom |
| **Trade Balance Waterfall** | Exports, Imports, Net Trade | Track global integration |
| **Policy Impact Waterfall** | Event-wise growth decomposition | Attribute growth to policies |
| **Correlation Heatmap** | Inter-indicator relationships | Understand economic linkages |

---

## Period Summary Statistics

### UPA Era (2004-2014)
| Metric | Value |
|--------|-------|
| Average GDP Growth | 7.8% |
| Average Inflation | 8.1% |
| Forex Reserves Change | +$163B |
| Cumulative FDI | $303B |

### NDA Era (2014-2024)
| Metric | Value |
|--------|-------|
| Average GDP Growth | 6.1%* |
| Average Inflation | 5.0% |
| Forex Reserves Change | +$341B |
| Cumulative FDI | $665B |

*Note: Includes COVID-19 impact year (-6.6% in 2020-21)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/MAYANK12-WQ/India-Economic-Comparison.git
cd India-Economic-Comparison

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

---

## Quick Start

### Generate All Charts
```bash
python generate_charts.py
python generate_advanced_charts.py
```

### View Interactive Dashboard
Open `charts/index.html` in your browser.

### Run Full Comparison Analysis
```python
from src.comparison_engine import ComparisonEngine

engine = ComparisonEngine()
result = engine.run_full_comparison()
print(result.executive_summary)
```

### Command Line Interface
```bash
# Compare all indicators
india-econ compare --output report.json

# Analyze specific policy impact
india-econ causal --policy demonetization --method did

# Run counterfactual scenario
india-econ counterfactual --scenario no_demonetization

# Get debate response with data backing
india-econ debate "UPA benefited from favorable global conditions"
```

---

## Project Architecture

```
India-Economic-Comparison/
├── charts/                      # Generated interactive visualizations
│   ├── index.html              # Main dashboard
│   ├── 01_gdp_growth_timeseries.html
│   ├── 02_period_comparison_bars.html
│   ├── ...
│   └── 14_statistical_summary.html
├── src/
│   ├── comparison_engine.py    # Main orchestration engine
│   ├── cli.py                  # Command line interface
│   ├── data_pipeline/          # Data fetching, validation, caching
│   ├── data_skepticism/        # Quality assessment, bias detection
│   ├── causal_inference/       # DiD, Synthetic Control, RDD
│   ├── counterfactual/         # What-if scenario simulation
│   ├── uncertainty/            # Bootstrap confidence intervals
│   ├── debate_assistant/       # Real-time argument responses
│   ├── ethical_framework/      # Balanced analysis enforcement
│   ├── predictive_forensics/   # Political claim validation
│   ├── visualization/          # Chart generation
│   └── api/                    # REST API endpoints
├── data/
│   └── sample_indicators.json  # Economic data 2004-2024
├── tests/                      # Comprehensive test suite
├── config/
│   └── settings.yaml           # Configuration
├── generate_charts.py          # Basic chart generator
├── generate_advanced_charts.py # Advanced economist-grade charts
└── requirements.txt
```

---

## Analytical Methodology

### Causal Inference Methods
- **Difference-in-Differences (DiD)**: Compare treatment effects before/after policy interventions
- **Synthetic Control Method**: Construct counterfactual scenarios
- **Regression Discontinuity Design**: Analyze threshold-based policy effects

### Uncertainty Quantification
- Bootstrap confidence intervals (10,000 iterations)
- Bayesian credible intervals
- Sensitivity analysis for key assumptions

### Data Quality Framework
- Multi-source cross-validation
- Methodology change adjustments
- Bias detection and correction
- Reliability scoring (0-10 scale)

---

## Ethical Principles

This project adheres to strict ethical guidelines for economic analysis:

| Principle | Implementation |
|-----------|---------------|
| **No Cherry-Picking** | All time periods 2004-2024 included |
| **Equal Scrutiny** | Same rigor applied to both periods |
| **Transparent Assumptions** | All methodological choices documented |
| **Acknowledge Limitations** | Data quality issues clearly stated |
| **No Oversimplification** | No single "winner" declared |
| **Avoid Hindsight Bias** | Decisions judged on contemporaneous info |
| **Human Impact First** | Welfare outcomes prioritized |

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/indicators` | GET | List available indicators |
| `/api/v1/compare` | POST | Compare two periods |
| `/api/v1/causal/analyze` | POST | Run causal analysis |
| `/api/v1/counterfactual/simulate` | POST | Simulate alternative scenario |
| `/api/v1/debate/respond` | POST | Get data-backed debate response |
| `/api/v1/claims/validate` | POST | Validate political claims |

### Python API

```python
from src.comparison_engine import ComparisonEngine

# Initialize
engine = ComparisonEngine()

# Run comparison
result = engine.run_full_comparison()

# Access specific comparisons
gdp_comp = result.indicator_comparisons["gdp_growth_rate"]
print(f"UPA Mean: {gdp_comp['period_a']['mean']:.1f}%")
print(f"NDA Mean: {gdp_comp['period_b']['mean']:.1f}%")
print(f"Statistically Significant: {gdp_comp['significant']}")

# Causal analysis
causal = engine.causal_engine.analyze_policy_impact(
    "demonetization", method="did"
)
print(causal.interpretation)

# Debate response
response = engine.get_debate_response(
    "GDP growth was higher under UPA"
)
print(response.response_text)
```

---

## Data Sources

| Source | Indicators | Frequency |
|--------|-----------|-----------|
| **MOSPI** | GDP, GVA, Per Capita Income | Annual/Quarterly |
| **RBI** | Inflation, Forex, Current Account | Monthly/Annual |
| **World Bank** | Development indicators, PPP data | Annual |
| **IMF** | WEO projections, fiscal data | Annual |
| **DPIIT** | FDI inflows by sector | Monthly |
| **CMIE** | Unemployment, LFPR | Monthly |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Run tests (`pytest tests/`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup
```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

---

## Citation

If you use this system in research or publications:

```bibtex
@software{india_economic_comparison_2024,
  title = {India Economic Comparison System},
  author = {Economic Analysis Team},
  year = {2024},
  url = {https://github.com/MAYANK12-WQ/India-Economic-Comparison},
  description = {Comprehensive analysis platform for comparing India's
                 economic performance across UPA (2004-2014) and
                 NDA (2014-2024) periods}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This system is designed for **educational and research purposes**. Economic analysis is inherently complex, and no comparison can fully capture all relevant factors.

**Users should:**
- Cross-validate findings with other sources
- Consider the documented methodological limitations
- Avoid drawing oversimplified conclusions
- Use the provided uncertainty intervals
- Acknowledge that external conditions differed significantly between periods

**This system aims to inform debate, not settle it.**

---

<p align="center">
  <strong>Built with Python, Plotly, and Rigorous Economic Methodology</strong>
</p>

<p align="center">
  Data Sources: MOSPI | RBI | World Bank | IMF | DPIIT | CMIE
</p>

<p align="center">
  <a href="charts/index.html">View Interactive Dashboard</a>
</p>
