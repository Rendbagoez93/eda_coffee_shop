# Coffee Sales Analysis - Analysis Modules

This directory contains modular analysis components aligned with the ARD (Analysis Requirements Document).

## 📊 Analysis Modules

### 1. Sales Performance Analysis (`sales_performance.py`)
**ARD Section 4.1** - Revenue Optimization

**Business Questions:**
- How much total revenue is generated?
- Which coffee products contribute the most/least revenue?
- How does revenue change over time?

**Analyses:**
- Total revenue and transaction metrics
- Revenue by product with performance tiers
- Daily, weekday, and monthly revenue trends
- Price distribution by product

**Key Outputs:**
- `product_performance_detailed.csv`
- `daily_revenue.csv`
- `price_by_coffee.csv`
- `sales_summary_metrics.json`

---

### 2. Time-Based Demand Analysis (`time_demand.py`)
**ARD Section 4.2** - Operational Efficiency

**Business Questions:**
- What are the peak and off-peak hours?
- How does demand differ by time of day?
- Are there weekday vs weekend differences?

**Analyses:**
- Hourly demand patterns with peak hour identification
- Time of day segment analysis
- Weekday patterns
- Hour-weekday heatmap data
- Peak vs off-peak comparison
- Weekend vs weekday comparison

**Key Outputs:**
- `hourly_detailed_analysis.csv`
- `hour_weekday_heatmap.csv`
- `peak_offpeak_comparison.json`
- `weekend_weekday_comparison.json`

---

### 3. Product Preference Analysis (`product_preference.py`)
**ARD Section 4.3** - Product Strategy

**Business Questions:**
- Which coffee products are most popular?
- Do product preferences vary by time or day?
- Are certain products time-specific?

**Analyses:**
- Product popularity and market share
- Product performance by time of day
- Product patterns by hour (top 10)
- Product preferences by weekday
- Time-specific product identification

**Key Outputs:**
- `coffee_counts.csv`
- `product_time_of_day_analysis.csv`
- `product_hourly_patterns.csv`
- `product_weekday_patterns.csv`
- `time_specific_products.csv`

---

### 4. Payment Behavior Analysis (`payment_behavior.py`)
**ARD Section 4.4** - Customer Behavior

**Business Questions:**
- What payment methods do customers prefer?
- Does payment type influence spending value?
- Does payment behavior vary by time?

**Analyses:**
- Payment method distribution
- Spending patterns by payment type
- Payment preferences by hour and weekday
- Payment methods by time segment
- Cash vs cashless trends

**Key Outputs:**
- `payment_distribution.csv`
- `payment_hourly_patterns.csv`
- `payment_weekday_patterns.csv`
- `spending_by_payment.json`
- `cash_cashless_comparison.json`

---

### 5. Seasonality & Calendar Analysis (`seasonality.py`)
**ARD Section 4.5** - Forecasting & Planning

**Business Questions:**
- Do sales fluctuate by month?
- Are there identifiable seasonal patterns?

**Analyses:**
- Monthly revenue trends with MoM growth
- Seasonal pattern identification
- Quarterly performance analysis
- Day of month patterns
- Overall growth trends

**Key Outputs:**
- `monthly_trends_analysis.csv`
- `quarterly_performance.csv`
- `seasonal_patterns.json`
- `growth_trends.json`

---

## 🎯 Main Orchestrator (`analyzer.py`)

The `CoffeeSalesAnalyzer` class coordinates all analysis modules and provides:
- Unified interface for running all analyses
- Consolidated insight generation
- Executive summary creation
- Summary statistics across all analyses

## 🚀 Usage

### Run Individual Analysis
```python
from src.analysis import SalesPerformanceAnalyzer
import pandas as pd

df = pd.read_csv('data/enriched/enriched_coffee_sales.csv')
analyzer = SalesPerformanceAnalyzer()
results = analyzer.run_analysis(df)
insights = analyzer.get_insights()
```

### Run Complete Analysis Suite
```python
from src.analysis import CoffeeSalesAnalyzer
import pandas as pd

df = pd.read_csv('data/enriched/enriched_coffee_sales.csv')
analyzer = CoffeeSalesAnalyzer()
results = analyzer.run_all_analyses(df)
analyzer.print_insights()
```

### Command Line
```bash
# Run complete analysis pipeline
python -m src.analysis.run_analysis

# Or use the test script
python test_complete_pipeline.py
```

## 📈 Output Structure

All analysis outputs are saved to the `outputs/` directory:

```
outputs/
├── executive_summary.txt          # High-level summary
├── executive_summary.json         # Programmatic summary
├── sales_summary_metrics.json     # Revenue metrics
├── product_performance_detailed.csv
├── daily_revenue.csv
├── hourly_detailed_analysis.csv
├── payment_distribution.csv
├── monthly_trends_analysis.csv
└── ... (additional analysis files)
```

## 🎨 Key Features

✅ **Modular Design** - Each analysis is self-contained and independent  
✅ **ARD-Aligned** - Direct mapping to business requirements  
✅ **Configurable** - All analyses can be enabled/disabled via config  
✅ **Robust** - Comprehensive error handling and logging  
✅ **Insightful** - Automatic business insight generation  
✅ **Well-Documented** - Clear docstrings and comments  

## 📝 Business Impact

Each analysis module directly supports the three main business objectives:

1. **Revenue Optimization** - Sales performance and product insights
2. **Operational Efficiency** - Time-based demand patterns
3. **Product & Marketing Strategy** - Product preferences and customer behavior

---

*For complete project documentation, see ARD.md in the project root.*
