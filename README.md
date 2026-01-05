# Coffee Shop Sales Analysis

Comprehensive exploratory data analysis (EDA) and business intelligence platform for coffee shop sales data. This project provides end-to-end analytics pipeline from data preprocessing to interactive visualizations and dashboards.

## 🎯 Project Overview

This project analyzes transaction-level coffee shop data to uncover:
- **Sales Performance**: Revenue trends, patterns, and growth analysis
- **Time-Based Demand**: Hourly patterns, peak hours, and demand forecasting
- **Product Preferences**: Popular products, pricing analysis, and performance metrics
- **Payment Behavior**: Payment method trends and spending patterns
- **Seasonality**: Monthly trends, seasonal patterns, and temporal variations

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
```powershell
git clone <repository-url>
cd eda_coffee_shop
```

2. **Install dependencies using UV** (recommended)
```powershell
uv sync
```

Or using pip:
```powershell
pip install -e .
```

### Running the Analysis

**Complete Pipeline** (preprocessing → analysis → visualization):
```powershell
uv run python main.py
uv run python run_visualizations.py
```

**Skip preprocessing** if data already processed:
```powershell
uv run python main.py --skip-preprocessing
uv run python run_visualizations.py
```

**Generate specific visualizations**:
```powershell
# Only sales and time modules
uv run python run_visualizations.py --module sales,time

# Skip interactive dashboard
uv run python run_visualizations.py --skip-dashboard
```

## Visualization and Dashboard Generation

The project includes a comprehensive visualization module that generates professional, publication-ready charts and interactive dashboards.

### Quick Start - Visualizations

1. **Run the complete analysis pipeline first** (to generate data):
```powershell
python main.py
```

2. **Generate all visualizations**:
```powershell
python run_visualizations.py
```

3. **View results** in the `output/` directory:
   - Static PNG images for all analysis areas
   - Interactive HTML dashboard
   - Visualization index (catalog of all files)

### Visualization Modules

The visualization system includes 5 specialized modules plus dashboards:

1. **Sales Performance** (`sales_viz.py`)
   - Revenue trends and analysis
   - Weekday performance comparison
   - Price distribution analysis

2. **Time-Based Demand** (`time_viz.py`)
   - Hourly demand patterns
   - Hour × Weekday heatmaps
   - Peak vs off-peak analysis

3. **Product Preference** (`product_viz.py`)
   - Product popularity rankings
   - Time-of-day patterns
   - Product metrics and pricing

4. **Payment Behavior** (`payment_viz.py`)
   - Payment method distribution
   - Spending patterns
   - Payment trends over time

5. **Seasonality & Trends** (`seasonality_viz.py`)
   - Monthly revenue trends
   - Seasonal patterns
   - Growth analysis

6. **Dashboards** (`dashboard.py`)
   - Executive summary (static PNG)
   - Interactive dashboard (HTML with Plotly)
   - Visualization index

### Generate Specific Visualizations

```powershell
# Only sales and time visualizations
python run_visualizations.py --module sales,time

# Skip interactive dashboard
python run_visualizations.py --skip-dashboard
```

### Testing Visualizations

```powershell
python test_visualizations.py
```

### Visualization Documentation

For detailed documentation on the visualization module, see:
- `src/visualization/README.md` - Complete visualization guide
- Generated `output/VISUALIZATION_INDEX.md` - Catalog of all visualizations

## Complete Pipeline

The project now includes a complete, modular pipeline:

1. **Configuration** - YAML-based settings (`config/`)
2. **Data Processing** - Preprocessing and enrichment (`src/data/`)
3. **Analysis** - 5 comprehensive analysis modules (`src/analysis/`)
4. **Visualization** - Charts and dashboards (`src/visualization/`)
5. **Pipeline Orchestration** - Unified execution (`main.py`)

### Run Complete Pipeline

```powershell
# Full pipeline (preprocessing → enrichment → analysis → visualization)
python main.py
python run_visualizations.py

# Or skip preprocessing if data already processed
python main.py --skip-preprocessing
python run_visualizations.py
```

## 📁 Project Structure

```
eda_coffee_shop/
├── config/                          # Configuration files
│   ├── config.yaml                 # Main configuration
│   ├── paths.yaml                  # File paths
│   └── logging.yaml                # Logging configuration
├── data/
│   ├── raw/                        # Raw data files
│   │   └── Coffe_sales.csv
│   └── enriched/                   # Processed and enriched data
│       ├── preprocessed_data.csv
│       └── enriched_coffee_sales.csv
├── src/
│   ├── data/                       # Data processing modules
│   │   ├── preprocessor.py         # Data cleaning and validation
│   │   ├── enrichment.py           # Feature engineering
│   │   └── pipeline.py             # Data pipeline orchestration
│   ├── analysis/                   # Analysis modules (5 modules)
│   │   ├── sales_performance.py    # Sales and revenue analysis
│   │   ├── time_demand.py          # Time-based demand patterns
│   │   ├── product_preference.py   # Product performance analysis
│   │   ├── payment_behavior.py     # Payment method analysis
│   │   ├── seasonality.py          # Seasonal trends analysis
│   │   ├── analyzer.py             # Main analyzer orchestrator
│   │   └── run_analysis.py         # Analysis runner
│   ├── visualization/              # Visualization modules (6 modules)
│   │   ├── base_viz.py             # Base visualizer class
│   │   ├── sales_viz.py            # Sales visualizations
│   │   ├── time_viz.py             # Time-based visualizations
│   │   ├── product_viz.py          # Product visualizations
│   │   ├── payment_viz.py          # Payment visualizations
│   │   ├── seasonality_viz.py      # Seasonality visualizations
│   │   └── dashboard.py            # Interactive dashboard generator
│   └── utils/                      # Utility modules
│       ├── config_loader.py        # YAML configuration loader
│       └── logger.py               # Logging utilities
├── docs/                           # Documentation
│   ├── ARD-COFFEE-SALES.md         # Architecture decision records
│   ├── ANALYSIS-MODULE.md          # Analysis module documentation
│   └── IMPLEMENTATION_GUIDE.md     # Implementation guide
├── notebooks/                      # Jupyter notebooks
│   └── eda-analysis-on-daily-coffee-transaction.ipynb
├── output/                         # Generated analysis outputs
│   ├── *.csv                       # Analysis result tables
│   ├── *.json                      # Summary metrics and insights
│   ├── *.png                       # Static visualizations
│   ├── dashboard_interactive.html  # Interactive HTML dashboard
│   ├── visualizations/             # Organized visualization files
│   └── VISUALIZATION_INDEX.md      # Catalog of all visualizations
├── tests/                          # Test files
│   ├── test_complete_pipeline.py   # Integration tests
│   ├── test_modules.py             # Module unit tests
│   └── test_visualizations.py      # Visualization tests
├── main.py                         # Main pipeline orchestrator
├── run_visualizations.py           # Visualization runner script
├── pyproject.toml                  # Project dependencies (UV/pip)
├── VISUALIZATION_QUICKSTART.md     # Quick start for visualizations
└── VISUALIZATION_IMPLEMENTATION.md # Visualization implementation details
```

## 📦 Dependencies

Defined in [pyproject.toml](pyproject.toml):

### Core Analysis
- **pandas** >= 2.3.3 - Data manipulation and analysis
- **scipy** >= 1.16.3 - Scientific computing
- **scikit-learn** >= 1.7.2 - Machine learning utilities
- **pyyaml** >= 6.0.3 - Configuration management

### Visualization
- **matplotlib** >= 3.10.6 - Static plotting
- **seaborn** >= 0.13.2 - Statistical visualizations
- **plotly** >= 5.18.0 - Interactive dashboards
- **kaleido** >= 0.2.1 - Static export from Plotly

### Data Source
- **kaggle** >= 1.7.4.5 - Kaggle API client
- **kagglehub** >= 0.3.13 - Kaggle dataset management

### Installation

**Using UV** (recommended):
```powershell
uv sync
```

**Using pip**:
```powershell
pip install -e .
```

## 🧪 Testing

```powershell
# Test complete pipeline
uv run python test_complete_pipeline.py

# Test individual modules
uv run python test_modules.py

# Test visualizations
uv run python test_visualizations.py
```

## 📊 Output Files

After running the complete pipeline, the `output/` directory contains:

### Data Analysis Results
- CSV files with detailed analysis tables
- JSON files with summary metrics and insights

### Visualizations
- PNG files for all static charts (sales, time, product, payment, seasonality)
- Interactive HTML dashboard (`dashboard_interactive.html`)
- Visualization index catalog (`VISUALIZATION_INDEX.md`)

### Key Outputs
- `executive_summary.json` - High-level business insights
- `executive_summary.txt` - Text-based summary report
- `sales_summary_metrics.json` - Sales KPIs
- `growth_trends.json` - Growth and trend analysis

## 📚 Documentation

- [src/visualization/README.md](src/visualization/README.md) - Visualization module guide
- [VISUALIZATION_QUICKSTART.md](VISUALIZATION_QUICKSTART.md) - Quick start guide
- [VISUALIZATION_IMPLEMENTATION.md](VISUALIZATION_IMPLEMENTATION.md) - Implementation details
- [docs/ARD-COFFEE-SALES.md](docs/ARD-COFFEE-SALES.md) - Architecture decisions
- [docs/ANALYSIS-MODULE.md](docs/ANALYSIS-MODULE.md) - Analysis module docs
- [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Implementation guide

## 🚧 Future Enhancements

- [ ] Advanced time series forecasting (ARIMA, Prophet)
- [ ] Customer segmentation with clustering
- [ ] Anomaly detection for unusual patterns
- [ ] Automated PDF report generation
- [ ] Real-time dashboard with live data
- [ ] A/B testing framework
- [ ] Recommendation system for products

---

## 📄 License

This project is for educational and analytical purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows existing structure and patterns
2. All tests pass
3. Documentation is updated

---

**Project Version**: 2.0  
**Last Updated**: January 2026  
**Status**: ✅ Complete analysis and visualization pipeline  
**Python**: 3.13+  
**Package Manager**: UV
