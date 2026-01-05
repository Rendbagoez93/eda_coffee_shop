# Visualization Module

This module provides comprehensive visualization and dashboard generation for the Coffee Shop Sales Analysis project.

## Overview

The visualization module creates professional, publication-ready visualizations for all analysis areas defined in the ARD document:

1. **Sales Performance** - Revenue metrics, product performance, daily/weekday analysis
2. **Time-Based Demand** - Hourly patterns, peak hours, heatmaps, weekend vs weekday
3. **Product Preference** - Product popularity, pricing, time-of-day patterns
4. **Payment Behavior** - Payment methods, spending patterns, trends
5. **Seasonality & Trends** - Monthly trends, seasonal patterns, growth analysis
6. **Dashboards** - Executive summary and interactive dashboards

## Module Structure

```
src/visualization/
├── __init__.py              # Package initialization
├── base_viz.py              # Base visualizer with common utilities
├── sales_viz.py             # Sales performance visualizations
├── time_viz.py              # Time-based demand visualizations
├── product_viz.py           # Product preference visualizations
├── payment_viz.py           # Payment behavior visualizations
├── seasonality_viz.py       # Seasonality and trend visualizations
└── dashboard.py             # Dashboard generator
```

## Quick Start

### Generate All Visualizations

```bash
python run_visualizations.py
```

### Generate Specific Modules

```bash
# Only sales and time visualizations
python run_visualizations.py --module sales,time

# Skip dashboard generation
python run_visualizations.py --skip-dashboard
```

### Programmatic Usage

```python
from src.visualization.dashboard import DashboardGenerator

# Generate all visualizations
dashboard = DashboardGenerator()
results = dashboard.generate_all_visualizations()

# Create visualization index
dashboard.create_visualization_index()
```

## Visualization Modules

### 1. Sales Performance (sales_viz.py)

**Visualizations:**
- Revenue overview (daily/weekday trends)
- Weekday analysis (revenue and transaction distribution)
- Price analysis (price distribution, coffee pricing)

**Key Methods:**
- `plot_revenue_overview()` - Overall revenue trends
- `plot_weekday_analysis()` - Weekday performance comparison
- `plot_price_analysis()` - Pricing and distribution analysis

**Output Files:**
- `sales_revenue_overview.png`
- `sales_weekday_analysis.png`
- `sales_price_analysis.png`

### 2. Time-Based Demand (time_viz.py)

**Visualizations:**
- Hourly demand patterns (revenue and transactions by hour)
- Hour-weekday heatmap (demand patterns across time)
- Peak vs off-peak analysis (period comparisons)

**Key Methods:**
- `plot_hourly_demand()` - Hour-by-hour analysis
- `plot_heatmap()` - Hour × Weekday heatmap
- `plot_peak_analysis()` - Peak/off-peak and weekend/weekday comparisons

**Output Files:**
- `time_hourly_demand.png`
- `time_demand_heatmap.png`
- `time_peak_analysis.png`

### 3. Product Preference (product_viz.py)

**Visualizations:**
- Product popularity (transaction count, revenue, pricing, market share)
- Time-of-day patterns (product preferences by time period)
- Hourly patterns (product-hour heatmap)
- Product metrics (price vs popularity, Pareto analysis, distribution)

**Key Methods:**
- `plot_product_popularity()` - Product ranking and market share
- `plot_time_of_day_patterns()` - Time-based product preferences
- `plot_hourly_patterns()` - Product demand by hour heatmap
- `plot_product_metrics()` - Comprehensive product analysis

**Output Files:**
- `product_popularity.png`
- `product_time_patterns.png`
- `product_hourly_heatmap.png`
- `product_metrics.png`

### 4. Payment Behavior (payment_viz.py)

**Visualizations:**
- Payment distribution (method breakdown and metrics)
- Spending patterns (revenue, average transaction, cash vs cashless)
- Payment trends (hourly and weekday patterns)
- Payment-product relationship (heatmap and distribution by product)

**Key Methods:**
- `plot_payment_distribution()` - Payment method breakdown
- `plot_spending_patterns()` - Spending analysis by payment method
- `plot_payment_trends()` - Temporal payment patterns
- `plot_payment_product_relationship()` - Payment × Product analysis

**Output Files:**
- `payment_distribution.png`
- `payment_spending_patterns.png`
- `payment_trends.png`
- `payment_product_relationship.png`

### 5. Seasonality & Trends (seasonality_viz.py)

**Visualizations:**
- Monthly trends (revenue, transactions, growth, cumulative)
- Seasonal patterns (by season, by weekday, variance analysis)
- Growth analysis (growth metrics, trends, month-over-month changes)

**Key Methods:**
- `plot_monthly_trends()` - Monthly performance analysis
- `plot_seasonal_patterns()` - Seasonal and day-of-week patterns
- `plot_growth_analysis()` - Growth trends and forecasting

**Output Files:**
- `seasonality_monthly_trends.png`
- `seasonality_patterns.png`
- `seasonality_growth.png`

### 6. Dashboard (dashboard.py)

**Features:**
- Executive summary dashboard (key metrics overview)
- Interactive Plotly dashboard (HTML with interactive charts)
- Visualization index (catalog of all generated files)
- Batch generation of all visualizations

**Key Methods:**
- `create_executive_summary()` - Static summary dashboard
- `create_interactive_dashboard()` - Interactive HTML dashboard
- `generate_all_visualizations()` - Run all visualization modules
- `create_visualization_index()` - Generate file catalog

**Output Files:**
- `dashboard_executive_summary.png`
- `dashboard_interactive.html`
- `VISUALIZATION_INDEX.md`

## Base Visualizer (base_viz.py)

Common utilities for all visualizers:

**Features:**
- Consistent styling and color schemes
- Data loading utilities (CSV and JSON)
- Figure saving with automatic directory creation
- Grid layout creation
- Number and currency formatting

**Key Methods:**
- `setup_style()` - Configure matplotlib style
- `load_csv_data(filename)` - Load CSV from output directory
- `load_json_data(filename)` - Load JSON from output directory
- `save_figure(fig, filename)` - Save figure to output directory
- `create_grid_layout(rows, cols)` - Create subplot grid
- `format_number(num)` - Format numbers with K/M suffixes
- `format_currency(amount)` - Format currency values

## Configuration

Visualizations read from the same configuration files as the analysis pipeline:

**paths.yaml:**
```yaml
output: "output"
```

**config.yaml:**
```yaml
visualization:
  dpi: 300
  style: seaborn-v0_8-darkgrid
  figsize: [16, 10]
  save_format: png
```

## Dependencies

**Required:**
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0
- numpy >= 1.24.0

**Optional (for interactive dashboards):**
- plotly >= 5.18.0
- kaleido >= 0.2.1

Install all dependencies:
```bash
pip install matplotlib seaborn pandas numpy plotly kaleido
```

## Output

All visualizations are saved to the `output/` directory configured in `paths.yaml`.

**File naming convention:**
- `{module}_{description}.png` - Static visualizations
- `dashboard_*.png` - Dashboard images
- `dashboard_interactive.html` - Interactive dashboard
- `VISUALIZATION_INDEX.md` - File catalog

## Examples

### Example 1: Generate Only Sales Visualizations

```python
from src.visualization.sales_viz import SalesVisualizer

sales_viz = SalesVisualizer()
files = sales_viz.create_all_visualizations()

for file in files:
    print(f"Created: {file}")
```

### Example 2: Create Custom Visualization

```python
from src.visualization.base_viz import BaseVisualizer
import matplotlib.pyplot as plt

class CustomVisualizer(BaseVisualizer):
    def plot_custom(self):
        # Load data
        data = self.load_csv_data('my_data.csv')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot
        ax.plot(data['x'], data['y'])
        ax.set_title('Custom Plot')
        
        # Save
        return self.save_figure(fig, 'custom_plot.png')

viz = CustomVisualizer()
viz.plot_custom()
```

### Example 3: Generate Dashboard Only

```python
from src.visualization.dashboard import DashboardGenerator

dashboard = DashboardGenerator()

# Executive summary
summary = dashboard.create_executive_summary()
print(f"Summary: {summary}")

# Interactive dashboard
interactive = dashboard.create_interactive_dashboard()
print(f"Interactive: {interactive}")

# Index
index = dashboard.create_visualization_index()
print(f"Index: {index}")
```

## Best Practices

1. **Run Analysis First**: Ensure analysis pipeline has run and generated output files
2. **Check Dependencies**: Install plotly for interactive dashboards
3. **Review Configuration**: Verify paths.yaml points to correct output directory
4. **Modular Approach**: Generate specific modules for faster iteration
5. **Batch Generation**: Use dashboard.generate_all_visualizations() for complete suite

## Troubleshooting

**Issue: Missing data files**
```
Solution: Run main.py first to generate analysis results
python main.py
```

**Issue: Plotly not available**
```
Solution: Install plotly and kaleido
pip install plotly kaleido
```

**Issue: Figures not saving**
```
Solution: Check output directory exists and is writable
Verify paths.yaml configuration
```

**Issue: Poor image quality**
```
Solution: Increase DPI in config.yaml
visualization:
  dpi: 300  # or higher
```

## Integration with Main Pipeline

The visualization module can be integrated into the main analysis pipeline:

```python
# main.py (add at end)
from src.visualization.dashboard import DashboardGenerator

def main():
    # ... existing preprocessing, enrichment, analysis ...
    
    # Generate visualizations
    if args.generate_viz:
        logger.info("Generating visualizations...")
        dashboard = DashboardGenerator()
        dashboard.generate_all_visualizations()
        dashboard.create_visualization_index()
```

## Performance

Typical generation times (on standard hardware):

- Sales visualizations: ~2-3 seconds
- Time visualizations: ~2-3 seconds
- Product visualizations: ~3-4 seconds
- Payment visualizations: ~3-4 seconds
- Seasonality visualizations: ~3-4 seconds
- Executive dashboard: ~2-3 seconds
- Interactive dashboard: ~3-5 seconds
- **Total (all modules): ~20-30 seconds**

## Future Enhancements

Potential improvements:

1. **Advanced Interactivity**: Add filters and drill-down capabilities
2. **Animation**: Create animated trend visualizations
3. **Real-time Updates**: Support live data updates
4. **Custom Themes**: Add more color schemes and styles
5. **Export Formats**: Support PDF, SVG, and other formats
6. **Statistical Overlays**: Add confidence intervals, regression lines
7. **Comparative Analysis**: Year-over-year, month-over-month comparisons

## Support

For issues or questions:
1. Check this README
2. Review error logs in logs/ directory
3. Verify data files in output/ directory
4. Check configuration in config/ directory

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintainer**: Coffee Shop Analytics Team
