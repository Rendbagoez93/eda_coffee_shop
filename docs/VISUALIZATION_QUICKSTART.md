# Visualization Quick Reference Guide

## 📊 Quick Start

```bash
# 1. Run analysis pipeline (if not already done)
python main.py

# 2. Generate all visualizations
python run_visualizations.py

# 3. Check output/ directory for results
```

## 🎯 Common Tasks

### Generate All Visualizations
```bash
python run_visualizations.py
```

### Generate Specific Modules
```bash
# Sales only
python run_visualizations.py --module sales

# Sales and time only
python run_visualizations.py --module sales,time

# All except dashboard
python run_visualizations.py --skip-dashboard
```

### Test Visualizations
```bash
python test_visualizations.py
```

## 📁 Output Files

### Sales Performance (3 files)
- `sales_revenue_overview.png` - Daily and weekday revenue trends
- `sales_weekday_analysis.png` - Weekday performance comparison
- `sales_price_analysis.png` - Price distribution and coffee pricing

### Time-Based Demand (3 files)
- `time_hourly_demand.png` - Hour-by-hour patterns
- `time_demand_heatmap.png` - Hour × Weekday heatmap
- `time_peak_analysis.png` - Peak vs off-peak comparison

### Product Preference (4 files)
- `product_popularity.png` - Product rankings and market share
- `product_time_patterns.png` - Time-of-day preferences
- `product_hourly_heatmap.png` - Product × Hour heatmap
- `product_metrics.png` - Comprehensive metrics

### Payment Behavior (4 files)
- `payment_distribution.png` - Payment method breakdown
- `payment_spending_patterns.png` - Spending analysis
- `payment_trends.png` - Temporal patterns
- `payment_product_relationship.png` - Payment × Product analysis

### Seasonality & Trends (3 files)
- `seasonality_monthly_trends.png` - Monthly performance
- `seasonality_patterns.png` - Seasonal patterns
- `seasonality_growth.png` - Growth analysis

### Dashboards (3 files)
- `dashboard_executive_summary.png` - Executive overview
- `dashboard_interactive.html` - Interactive dashboard
- `VISUALIZATION_INDEX.md` - File catalog

**Total: 20 visualization files**

## 🔧 Programmatic Usage

### Generate All Visualizations
```python
from src.visualization.dashboard import DashboardGenerator

dashboard = DashboardGenerator()
results = dashboard.generate_all_visualizations()
```

### Generate Specific Module
```python
from src.visualization.sales_viz import SalesVisualizer

sales_viz = SalesVisualizer()
files = sales_viz.create_all_visualizations()
```

### Create Executive Summary Only
```python
from src.visualization.dashboard import DashboardGenerator

dashboard = DashboardGenerator()
summary = dashboard.create_executive_summary()
print(f"Summary saved to: {summary}")
```

### Create Interactive Dashboard Only
```python
from src.visualization.dashboard import DashboardGenerator

dashboard = DashboardGenerator()
interactive = dashboard.create_interactive_dashboard()
print(f"Interactive dashboard: {interactive}")
```

## 📊 Visualization Types by Module

### Sales Performance
| Visualization | Type | Description |
|--------------|------|-------------|
| Revenue Overview | Line + Bar | Daily revenue trend, weekday totals |
| Weekday Analysis | Bar | Revenue and transactions by weekday |
| Price Analysis | Histogram + Bar | Price distribution, coffee pricing |

### Time-Based Demand
| Visualization | Type | Description |
|--------------|------|-------------|
| Hourly Demand | Bar + Line | Revenue and transactions by hour |
| Heatmap | Heatmap | Hour × Weekday demand pattern |
| Peak Analysis | Bar (dual axis) | Peak/off-peak, weekend/weekday |

### Product Preference
| Visualization | Type | Description |
|--------------|------|-------------|
| Popularity | Bar + Pie | Transaction count, revenue, market share |
| Time Patterns | Bar + Area | Product preferences by time of day |
| Hourly Heatmap | Heatmap | Product × Hour demand matrix |
| Metrics | Scatter + Bar | Price vs popularity, Pareto analysis |

### Payment Behavior
| Visualization | Type | Description |
|--------------|------|-------------|
| Distribution | Pie + Bar | Payment method breakdown |
| Spending Patterns | Bar + Donut | Revenue, avg transaction by method |
| Trends | Line + Bar | Hourly and weekday patterns |
| Product Relationship | Heatmap + Bar | Payment × Product matrix |

### Seasonality & Trends
| Visualization | Type | Description |
|--------------|------|-------------|
| Monthly Trends | Bar + Line | Revenue, transactions, growth |
| Seasonal Patterns | Bar + Pie | Seasonal comparison, weekday analysis |
| Growth Analysis | Bar + Line | Growth metrics, trend lines |

### Dashboards
| Visualization | Type | Description |
|--------------|------|-------------|
| Executive Summary | Multi-panel | Key metrics overview (static) |
| Interactive Dashboard | Plotly | Interactive charts (HTML) |
| Index | Markdown | Catalog of all visualizations |

## ⚙️ Configuration

### config.yaml
```yaml
visualization:
  dpi: 300              # Image resolution
  style: seaborn-v0_8-darkgrid
  figsize: [16, 10]     # Default figure size
  save_format: png      # Output format
```

### paths.yaml
```yaml
output: "output"        # Output directory for visualizations
```

## 🔍 Troubleshooting

### Issue: Missing data files
**Solution**: Run analysis pipeline first
```bash
python main.py
```

### Issue: Plotly not installed
**Solution**: Install plotly and kaleido
```bash
pip install plotly kaleido
```

### Issue: Poor image quality
**Solution**: Increase DPI in config.yaml
```yaml
visualization:
  dpi: 600  # Higher quality
```

### Issue: Files not saving
**Solution**: Check output directory
```bash
# Windows PowerShell
if (!(Test-Path output)) { New-Item -ItemType Directory -Path output }
```

## 📈 Performance

Typical generation times:
- Single module: 2-4 seconds
- All modules: 20-30 seconds
- Interactive dashboard: 3-5 seconds

## 🎨 Customization

### Change Color Scheme
```python
# In base_viz.py, modify setup_style()
sns.set_palette("husl")  # or any seaborn palette
```

### Change Figure Size
```python
# In individual visualizer
fig, ax = plt.subplots(figsize=(20, 12))  # Custom size
```

### Add Custom Visualization
```python
from src.visualization.base_viz import BaseVisualizer
import matplotlib.pyplot as plt

class CustomViz(BaseVisualizer):
    def plot_custom(self):
        data = self.load_csv_data('my_data.csv')
        fig, ax = plt.subplots()
        ax.plot(data['x'], data['y'])
        return self.save_figure(fig, 'custom.png')

viz = CustomViz()
viz.plot_custom()
```

## 📚 Resources

- **Main README**: `README.md`
- **Visualization Guide**: `src/visualization/README.md`
- **Analysis Guide**: `src/analysis/README.md`
- **Visualization Index**: `output/VISUALIZATION_INDEX.md` (generated)

## 🚀 Best Practices

1. **Run analysis first** - Visualizations need data files
2. **Use batch generation** - `generate_all_visualizations()` is efficient
3. **Check logs** - Review `logs/` for errors
4. **Review index** - Check `VISUALIZATION_INDEX.md` for file list
5. **Modular approach** - Generate specific modules during development

## 📦 Dependencies

**Required:**
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0
- numpy >= 1.24.0

**Optional (for interactive dashboards):**
- plotly >= 5.18.0
- kaleido >= 0.2.1

Install all:
```bash
pip install matplotlib seaborn pandas numpy plotly kaleido
```

## 💡 Tips

- Open `dashboard_interactive.html` in web browser for interactive exploration
- PNG files are high-resolution (300 DPI) suitable for presentations
- Check `VISUALIZATION_INDEX.md` for quick file reference
- Use `--skip-dashboard` for faster iteration during development
- Customize visualizations by editing individual visualizer modules

---

**Last Updated**: 2024  
**Version**: 1.0
