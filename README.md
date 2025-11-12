# Coffee Shop Sales Analysis 🎓

An exploratory data analysis (EDA) project analyzing coffee shop sales data to uncover patterns, trends, and insights about customer behavior, product performance, and revenue drivers.

## Project Overview

This project analyzes a coffee shop's transactional data to understand:
- **Sales patterns** by time (hourly, daily, weekly)
- **Product performance** and customer preferences
- **Revenue drivers** and pricing strategies
- **Customer behavior** across payment methods and time periods

### Dataset
- **Source**: `data/Coffe_sales.csv`
- **Key columns**: Date, Time, Coffee Name, Money (price), Cash Type, Hour of Day, Weekday, Time of Day, Month
- **Focus**: Transaction-level data capturing what customers buy, when, and how they pay

---

## Analysis Completed ✅

### 1. **Data Preparation & Cleaning**
- Standardized column names (lowercase, underscores)
- Converted date/time columns to proper datetime objects
- Categorized categorical variables (cash_type, coffee_name, time_of_day, weekday, month_name)
- Handled missing/coerced values with error handling
- Created derived features: day, month, year, weekend flag

### 2. **Descriptive Statistics**
- **Total Revenue**: Aggregated sales across all transactions
- **Average Transaction Value**: Mean spending per customer visit
- **Total Transactions**: Count of all sales
- **Top 5 Coffee Types**: Most purchased products
- **Payment Method Distribution**: Cash vs. card breakdown
- **Sales by Weekday**: Transaction counts per day of week

### 3. **Product Performance Analysis**
- **Coffee Price Summary** (`outputs/price_by_coffee.csv`)
  - Count, mean, std dev, min, max per coffee type
  - Identifies coffees with multiple price points (sizes, add-ons)
- **Sales Counts** (`outputs/coffee_counts.csv`)
  - Volume of sales per product
  - Identifies bestsellers vs. slow movers
- **Price Distribution by Product** (`outputs/boxplot_top_8.png`)
  - Visual comparison of top 8 coffees

### 4. **Time-Series & Trend Analysis**
- **Daily Revenue** (`outputs/daily_revenue.csv` & `outputs/daily_revenue.png`)
  - Revenue aggregated by date
  - Tracks revenue trends over time
- **Hourly Sales Patterns** (`outputs/hourly_counts.csv` & `outputs/sales_by_hour.png`)
  - Transaction counts by hour of day
  - Identifies peak vs. slow hours
  - Useful for staffing optimization
- **Weekday Revenue Analysis** (fixed & visualized)
  - Total revenue by day of week (Monday–Sunday)
  - Identifies high-performing vs. low-performing days

### 5. **Statistical Analysis**
- **Distribution Analysis**
  - Histogram & KDE of transaction amounts (`outputs/price_distribution.png`)
  - Skewness and kurtosis of money and hour_of_day
- **Outlier Detection**
  - Boxplot analysis to identify unusual transactions
- **Payment Method Analysis**
  - Revenue and transaction breakdown by payment type
- **Time-of-Day Analysis**
  - Statistical summaries (mean, median, std) by time period (morning, afternoon, evening, night)

### 6. **Correlation Analysis**
- **Numeric Correlations** (`outputs/numeric_correlations.csv`)
  - Correlation between hour_of_day, money, Weekdaysort, Monthsort
  - Identifies relationships between temporal variables and revenue
- **Heatmap Visualizations**
  - Pairwise correlations across time dimensions

---

## Current Outputs

The following files are generated in `/outputs/`:

| File | Type | Purpose |
|------|------|---------|
| `coffee_counts.csv` | CSV | Sales count per product |
| `price_by_coffee.csv` | CSV | Price statistics per product |
| `daily_revenue.csv` | CSV | Daily aggregated revenue |
| `hourly_counts.csv` | CSV | Transaction count by hour |
| `numeric_correlations.csv` | CSV | Correlation matrix |
| `daily_revenue.png` | Plot | Daily revenue trend line |
| `sales_by_hour.png` | Plot | Hourly sales bar chart |
| `price_distribution.png` | Plot | Histogram of transaction amounts |
| `boxplot_top_8.png` | Plot | Price distribution for top 8 coffees |

---

## Scripts

### `eda_coffee.py`
Modular, function-based EDA script. Cleaner structure with individual analysis functions:
- `load_data()` – Reads CSV
- `basic_summary()` – Data shape, types, missing values
- `sales_counts()` – Product popularity
- `price_by_coffee()` – Product pricing analysis
- `time_series_revenue()` – Daily trends
- `sales_by_hour()` – Hourly patterns
- `price_distribution()` – Transaction amount distribution
- `popular_coffees_boxplot()` – Top products price comparison
- `correlations()` – Numeric variable relationships
- `run_all()` – Execute all analyses

**Run**: `python eda_coffee.py`

### `coffee_sales_data_analysis.py`
Exploratory, script-based analysis with detailed prints and visualizations:
- Data loading and preprocessing
- Revenue overview metrics
- Category distributions
- Feature engineering
- Multiple chart visualizations (bar, line, hist, box, heatmap)
- Statistical summaries by groups (cash type, weekday, time of day)
- Fixed weekday plotting with proper ordering

**Run**: `python coffee_sales_data_analysis.py` (generates plot windows)

---

## Future Analysis Opportunities 🔮

### 1. **Advanced Time-Series Analysis**
- **Trend Decomposition**: Separate trend, seasonality, and residuals
- **Forecasting**: Use ARIMA, Prophet, or SARIMA to predict future revenue
- **Seasonal Patterns**: Analyze monthly/quarterly trends and holiday effects
- **Year-over-Year Comparison**: Compare same periods across years
- **Anomaly Detection**: Identify unusual spikes or drops in sales

### 2. **Customer Segmentation & Behavior**
- **RFM Analysis** (Recency, Frequency, Monetary):
  - Segment customers by visit patterns and spend
  - Identify VIP, at-risk, and inactive customers
- **Purchase Patterns**:
  - Identify products frequently bought together (market basket analysis)
  - Create association rules for cross-selling
- **Customer Lifetime Value (CLV)**:
  - Estimate long-term customer value
  - Tailor marketing to high-value segments

### 3. **Pricing & Revenue Optimization**
- **Price Elasticity Analysis**:
  - Measure how demand responds to price changes
  - Identify optimal price points per product
- **Revenue per Product Category**:
  - Margin analysis if cost data becomes available
  - Profit maximization strategies
- **A/B Testing Framework**:
  - Test price changes, promotions, menu changes
  - Measure uplift in revenue or transactions

### 4. **Marketing & Promotion Insights**
- **Peak Hour Optimization**:
  - Design promotions during slow hours
  - Bundle offers to balance demand
- **Promotional Effectiveness**:
  - If promotions are tracked, measure ROI
  - Identify best-performing promotions
- **Customer Retention**:
  - Loyalty program analysis if available
  - Win-back campaigns for lapsed customers

### 5. **Operational Improvements**
- **Staffing Optimization**:
  - Schedule staff based on hourly/daily demand
  - Predict busy periods and allocate resources
- **Inventory Management**:
  - Forecast demand per product
  - Reduce waste and optimize stock levels
- **Supplier & Cost Analysis**:
  - If costs are available, optimize procurement
  - Negotiate better terms based on volume patterns

### 6. **Advanced Visualizations & Dashboards**
- **Interactive Dashboards**:
  - Dash, Streamlit, or Power BI for dynamic exploration
  - Drill-down by date, product, time, payment method
- **Heatmaps**:
  - Hour × Day of Week heatmap for sales patterns
  - Product × Time-of-Day matrix
- **Geospatial Analysis** (if location data available):
  - Map sales by store location
  - Identify geographic hot spots

### 7. **Statistical Hypothesis Testing**
- **Comparative Analysis**:
  - Are weekday sales significantly different from weekend sales?
  - Do different payment methods correlate with different purchase amounts?
  - Is morning coffee demand different from evening?
- **ANOVA / T-tests**:
  - Test statistical significance of differences between groups
- **Chi-Square Tests**:
  - Test independence of categorical variables (e.g., coffee type vs. payment method)

### 8. **Machine Learning Models**
- **Sales Forecasting**:
  - Regression models to predict daily/hourly revenue
  - Multi-variable predictions considering multiple features
- **Product Recommendation System**:
  - Suggest products to customers based on past purchases
  - Content-based or collaborative filtering
- **Churn Prediction** (if customer IDs available):
  - Predict which customers are likely to stop buying
  - Target retention efforts

### 9. **Competitive & Market Analysis** (if external data available)
- **Benchmark Against Industry**:
  - Compare metrics to coffee shop industry standards
  - Identify competitive advantages
- **Market Trends**:
  - Track changing customer preferences
  - Respond to market demand shifts

### 10. **Text Analysis** (if available)
- **Customer Reviews/Feedback**:
  - Sentiment analysis
  - Topic modeling to identify pain points
  - Word frequency analysis

---

## Tech Stack

- **Python 3.13+**
- **pandas** – Data manipulation and aggregation
- **matplotlib** – Static visualizations
- **seaborn** – Statistical plots
- **numpy** – Numerical operations (optional in current usage)
- **kagglehub** (optional) – Data sourcing capabilities

### Dependencies
See `requirements.txt` or `pyproject.toml` for full dependency list.

---

## Getting Started

### Setup
```bash
# Create virtual environment (if using venv)
python -m venv .venv
.venv\Scripts\activate

# Or if using conda
conda activate your_env

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Run modular EDA
python eda_coffee.py

# Run comprehensive exploratory script (with interactive plots)
python coffee_sales_data_analysis.py
```

---

## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Transaction date |
| Time | time | Transaction time (HH:MM:SS) |
| Coffee_name | string | Product name |
| Money | float | Transaction amount |
| Cash_type | string | Payment method (e.g., Cash, Card) |
| Hour_of_day | int | Hour (0-23) |
| Weekday | string | Day of week (Monday–Sunday) |
| Time_of_day | string | Period (e.g., Morning, Afternoon) |
| Month_name | string | Month name |
| Weekdaysort | int | Numeric sort value for weekday |
| Monthsort | int | Numeric sort value for month |

---

## Key Insights (Findings So Far)

- ✅ Hourly patterns show peak demand during specific times
- ✅ Revenue varies significantly by day of week
- ✅ Top products drive majority of sales (Pareto principle)
- ✅ Multiple price points suggest product variants (sizes, add-ons)
- ✅ Payment methods show distribution across customer preferences

---

## Next Steps

1. **Implement forecasting** model to predict next week/month revenue
2. **Build interactive dashboard** for stakeholder reporting
3. **Perform A/B testing** on pricing or promotions
4. **Export findings** as a presentation/report
5. **Automate data pipeline** for regular updates

---

## Questions & Scope Refinement

To expand this analysis, consider:
- Are customer IDs available for cohort analysis?
- Do you have cost/profit margin data?
- Are there external events (holidays, promotions) to correlate?
- What is the business goal? (maximize revenue, customer retention, operational efficiency?)
- Is location data available (multiple stores)?

---

## License & Attribution

This analysis is part of a learning project on exploratory data analysis (EDA) for small business insights.

---

**Last Updated**: November 2025

For questions or contributions, please refer to the project repository.
