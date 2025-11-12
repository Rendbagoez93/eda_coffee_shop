# Coffee Shop Sales Analysis 🎓

A comprehensive exploratory data analysis (EDA) and advanced analytics project analyzing coffee shop sales data. Includes foundational EDA, time-series forecasting, customer segmentation, and anomaly detection to uncover patterns, trends, and actionable insights about customer behavior, product performance, and revenue optimization.

## Project Overview

This project analyzes a coffee shop's transactional data through multiple analytical lenses:
- **Exploratory Analysis**: Sales patterns by time (hourly, daily, weekly), product performance, customer preferences
- **Predictive Analytics**: Revenue forecasting using machine learning (Random Forest)
- **Customer Intelligence**: Behavior segmentation and customer journey mapping
- **Risk Analysis**: Anomaly detection and outlier identification
- **Business Optimization**: Strategic recommendations and operational insights

### Dataset
- **Source**: `data/Coffe_sales.csv`
- **Key columns**: Date, Time, Coffee Name, Money (price), Cash Type, Hour of Day, Weekday, Time of Day, Month
- **Focus**: Transaction-level data capturing what customers buy, when, and how they pay
- **Volume**: Thousands of transactions across multiple weeks/months

---

## Analysis Completed ✅

### Phase 0: Foundational EDA

#### 1. **Data Preparation & Cleaning**
- Standardized column names (lowercase, underscores)
- Converted date/time columns to proper datetime objects
- Categorized categorical variables (cash_type, coffee_name, time_of_day, weekday, month_name)
- Handled missing/coerced values with comprehensive error handling
- Created derived features: day, month, year, weekend flag, datetime composite

#### 2. **Descriptive Statistics & Business Overview**
- **Revenue Metrics**: Total revenue, average transaction value, transaction count
- **Product Performance**: Top 5 coffee types by volume and revenue
- **Customer Segmentation Basics**: Payment method distribution, cash type breakdown
- **Temporal Patterns**: Sales by weekday, hourly distribution, time-of-day breakdown
- **Weekend vs Weekday Comparison**: Transaction and revenue differences

#### 3. **Product Performance Analysis**
- **Coffee Price Summary** (`outputs/price_by_coffee.csv`)
  - Count, mean, std dev, min, max per coffee type
  - Identifies coffees with multiple price points (sizes, add-ons)
- **Sales Counts** (`outputs/coffee_counts.csv`)
  - Volume of sales per product
  - Identifies bestsellers vs. slow movers
- **Price Distribution by Product** (`outputs/boxplot_top_8.png`)
  - Visual comparison of top 8 coffees
- **Comprehensive Product Analysis** (`outputs/product_performance_detailed.csv`)
  - Orders, revenue, average price, peak day per product

#### 4. **Time-Series & Trend Analysis**
- **Daily Revenue Tracking** (`outputs/daily_revenue.csv` & `outputs/daily_revenue.png`)
  - Revenue aggregated by date
  - Trends and patterns over time
- **Hourly Sales Patterns** (`outputs/hourly_counts.csv`, `outputs/sales_by_hour.png`, `outputs/hourly_detailed_analysis.csv`)
  - Transaction counts and revenue by hour of day
  - Peak vs. slow hours identification
  - Staffing optimization insights
- **Weekday Revenue Analysis** (fixed & visualized)
  - Total revenue by day of week (Monday–Sunday)
  - Weekend vs. weekday comparison
  - Day-specific performance metrics
- **Comprehensive Dashboard** (`outputs/comprehensive_analysis.png`)
  - 4-panel visualization: hourly revenue, daily patterns, top products, distribution

#### 5. **Statistical Analysis**
- **Distribution Analysis**
  - Histogram with KDE of transaction amounts
  - Skewness and kurtosis calculations
  - Price range identification
  - Most common price points
- **Outlier Detection**
  - Boxplot analysis for unusual transactions
- **Payment Method Analysis**
  - Revenue breakdown by payment type
  - Transaction count by payment method
- **Time-of-Day Analysis**
  - Statistical summaries by time period (Morning, Afternoon, Evening, Night)
  - Mean, median, std dev per period

#### 6. **Correlation Analysis**
- **Numeric Correlations** (`outputs/numeric_correlations.csv`)
  - Correlation between hour_of_day, money, temporal variables
  - Identifies relationships affecting revenue
- **Heatmap Visualizations**
  - Pairwise correlation matrix across time and revenue dimensions

---

### Phase 1A: Time-Series Forecasting & Demand Prediction

#### **Advanced Analytics Using Machine Learning**
- **Random Forest Regression Model** for revenue prediction
- **Feature Engineering**: Day of year, day of week, month, weekend flag
- **Rolling Averages**: 7-day and 30-day moving averages for trend smoothing
- **Volatility Analysis**: 7-day rolling standard deviation
- **Model Performance Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- **Feature Importance Analysis**: Ranking of temporal features impact on revenue
- **7-Day Revenue Forecast**: Predictions for upcoming week with confidence
- **Seasonal Decomposition**: Trend vs. seasonal vs. residual components
- **Outputs**: 
  - `outputs/phase1_forecasting_results.csv` (historical + predicted data)
  - `outputs/phase1_forecasting_analysis.png` (4-panel forecast visualization)

---

### Phase 1B: Customer Segmentation & Behavior Analysis

#### **Advanced Customer Intelligence**
- **Time-Based Segmentation** (K-means clustering):
  - **4 Distinct Customer Segments** identified:
    1. High-Value Low-Volume hours (premium customers, fewer transactions)
    2. Peak Rush Hours (high volume, time-sensitive)
    3. Steady Customers (consistent, reliable demand)
    4. Off-Peak Hours (low activity, potential growth area)
- **Cluster Characteristics**:
  - Average transaction value per segment
  - Customer count distribution
  - Revenue contribution per segment
  - Top products per segment
- **Product Affinity Analysis**: Product preferences by time segment
- **Customer Journey Mapping**:
  - **Morning Rush (6-11 AM)**: Peak customer influx, coffee-focused
  - **Afternoon Peak (12-5 PM)**: Sustained demand, diverse products
  - **Evening Wind-down (6-10 PM)**: Declining volume, niche items
- **Behavioral Patterns**:
  - Hourly customer flow visualization
  - Day-of-week heatmap (activity intensity)
  - Segment profitability analysis
- **Outputs**:
  - `outputs/phase1_customer_segments.csv` (cluster assignments + metrics)
  - `outputs/phase1_customer_journey.csv` (hourly flow patterns)
  - `outputs/phase1_customer_segmentation.png` (6-panel segment visualization)

---

### Phase 1C: Anomaly Detection & Outlier Analysis

#### **Risk & Pattern Identification**
- **Daily Revenue Anomalies** (Z-score method):
  - Identifies unusually high/low revenue days
  - Separates weekend from weekday baselines
  - Flags days exceeding ±2 standard deviations
- **Transaction-Level Outliers** (IQR method):
  - Detects unusually high/low transaction amounts
  - Identifies top outlier transactions
  - Categorizes as high-value or low-value anomalies
- **Hourly Pattern Anomalies**:
  - Unusual hour-day combinations
  - Deviations from expected hourly patterns
  - Day-specific aberrations
- **Anomaly Insights**:
  - Percentage of anomalous days
  - Count and characteristics of outlier transactions
  - Monthly trend of anomalies
  - Potential business explanations (events, promotions, system errors)
- **Outputs**:
  - `outputs/phase1_daily_anomalies.csv` (daily revenue + z-scores)
  - `outputs/phase1_transaction_outliers.csv` (unusual transactions)
  - `outputs/phase1_anomaly_detection.png` (4-panel anomaly visualization)

---

## Current Outputs

### Phase 0 Outputs (Foundational EDA)
| File | Type | Purpose |
|------|------|---------|
| `coffee_counts.csv` | CSV | Sales count per product |
| `price_by_coffee.csv` | CSV | Price statistics per product |
| `product_performance_detailed.csv` | CSV | Detailed metrics per product |
| `daily_revenue.csv` | CSV | Daily aggregated revenue |
| `hourly_counts.csv` | CSV | Transaction count by hour |
| `hourly_detailed_analysis.csv` | CSV | Detailed hourly metrics |
| `daily_detailed_analysis.csv` | CSV | Detailed daily metrics |
| `numeric_correlations.csv` | CSV | Correlation matrix |
| `daily_revenue.png` | Plot | Daily revenue trend line |
| `sales_by_hour.png` | Plot | Hourly sales bar chart |
| `price_distribution.png` | Plot | Histogram of transaction amounts |
| `boxplot_top_8.png` | Plot | Price distribution for top 8 coffees |
| `comprehensive_analysis.png` | Plot | 4-panel foundational dashboard |

### Phase 1 Outputs (Advanced Analytics)
| File | Type | Purpose |
|------|------|---------|
| **Forecasting** | | |
| `phase1_forecasting_results.csv` | CSV | Historical + predicted daily revenue |
| `phase1_forecasting_analysis.png` | Plot | 4-panel forecasting dashboard (actual vs pred, trends, seasonal, volatility) |
| **Segmentation** | | |
| `phase1_customer_segments.csv` | CSV | Hour-based segments with cluster assignments |
| `phase1_customer_journey.csv` | CSV | Hourly customer flow patterns |
| `phase1_customer_segmentation.png` | Plot | 6-panel segmentation dashboard |
| **Anomalies** | | |
| `phase1_daily_anomalies.csv` | CSV | Daily revenue with z-scores and anomaly flags |
| `phase1_transaction_outliers.csv` | CSV | Unusual transactions identified |
| `phase1_anomaly_detection.png` | Plot | 4-panel anomaly visualization |

---

## Scripts & Execution

### `eda_coffee.py` – Modular Foundational EDA
Clean, function-based EDA script with individual analysis functions:
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

---

### `coffee_sales_data_analysis.py` – Exploratory Interactive Analysis
Detailed, script-based analysis with comprehensive prints and interactive visualizations:
- Data loading and preprocessing with column standardization
- Revenue overview metrics (total, average, transaction count)
- Category distributions and payment method breakdown
- Feature engineering (day, month, year, weekend flag)
- Revenue per hour and per day calculations
- Multiple chart visualizations (bar, line, histogram, boxplot, heatmap)
- Statistical summaries grouped by cash type, weekday, time of day
- Product performance ranking
- **Fixed weekday plotting** with proper ordering and missing day handling

**Run**: `python coffee_sales_data_analysis.py` (generates interactive plot windows)

---

### `basic_analysis.py` – Comprehensive Foundational Analytics
Advanced foundational EDA with structured output and strategic recommendations:
- Complete data loading and preprocessing
- Detailed business performance overview
- Temporal analysis: hourly, daily, weekly patterns
- Weekend vs. weekday comparative analysis
- Comprehensive 4-panel dashboard visualization
- Product performance detailed analysis
- Key business insights and recommendations
- Automatic CSV and PNG output generation
- Strategic recommendations (staffing, inventory, marketing, pricing, operations)

**Run**: `python basic_analysis.py`

**Outputs**: Creates multiple CSV files and PNG visualizations in `/outputs/`

---

### `run_phase1.py` – Advanced Machine Learning Analytics
Comprehensive Phase 1 analytics combining forecasting, segmentation, and anomaly detection:

#### Submodule: `time_series_forecasting(df)`
- Random Forest regression model for revenue prediction
- 80/20 train/test split
- Rolling averages (7-day, 30-day) for trend analysis
- Feature engineering (day of year, day of week, month, weekend flag)
- Model evaluation: MAE, RMSE, MAPE metrics
- Feature importance analysis
- Seasonal decomposition
- 7-day revenue forecast with daily predictions
- Visualization: actual vs. predicted, trends, seasonal patterns, volatility

#### Submodule: `customer_segmentation_analysis(df)`
- K-means clustering (4 clusters) on temporal-behavior features
- Customer segments: High-Value, Peak Rush, Steady, Off-Peak
- Product affinity analysis by segment
- Customer journey mapping: Morning Rush, Afternoon Peak, Evening Wind-down
- Hourly customer flow patterns
- Segment profitability and characteristics
- Multi-panel visualizations: scatter plots, pie charts, heatmaps, bar charts

#### Submodule: `anomaly_detection(df)`
- Z-score method for daily revenue anomalies
- IQR method for transaction-level outliers
- Weekend/weekday baseline comparison
- Hourly pattern anomaly detection
- Visualization: anomaly timeline, z-score distribution, boxplots
- Comprehensive outlier transaction analysis

**Run**: `python run_phase1.py`

**Outputs**: 9 files (3 CSV + 3 PNG per analysis + summary report)

---

### `eda-analysis-on-daily-coffee-transaction.ipynb` – Jupyter Notebook (Kaggle Format)
Interactive Jupyter notebook with exploratory analysis designed for Kaggle platform:
- Data loading from Kaggle input directory
- Data cleaning and preprocessing with robust error handling
- Business performance overview with markdown explanations
- Temporal analysis and customer behavior patterns
- Statistical analysis and distribution plots
- Interactive visualizations and findings

**Run**: `jupyter notebook eda-analysis-on-daily-coffee-transaction.ipynb`

**Note**: Currently shows pre-execution cells; ready to run in Jupyter environment

---

## Future Analysis Opportunities 🔮

### Phase 2: Business Optimization & Strategy

#### 1. **Pricing Strategy & Revenue Optimization**
- **Price Elasticity Analysis**:
  - Measure demand sensitivity to price changes
  - Identify optimal price points per product
  - Compare pricing across product categories
- **Dynamic Pricing Models**:
  - Time-of-day based pricing (premium during rush hours)
  - Day-of-week pricing adjustments
  - Seasonal pricing strategies
- **Bundle & Promotion Analysis**:
  - Identify high-margin product bundles
  - Model impact of volume discounts
  - Test promotion ROI if historical data available
- **Margin Optimization**:
  - If cost data available: profit maximization per product
  - Cost allocation strategies
  - Break-even analysis by product

#### 2. **Marketing & Customer Acquisition**
- **Customer Retention Analytics**:
  - If customer IDs available: repeat purchase patterns
  - Churn prediction models
  - Loyalty program design recommendations
- **Peak Hour Optimization**:
  - Design off-peak promotions to balance demand
  - Flash sales during low-demand periods
  - Loyalty rewards targeting slow hours
- **Product-Specific Marketing**:
  - Launch campaigns for underperforming products
  - Seasonal product recommendations
  - Cross-selling opportunities identification
- **Customer Lifetime Value (CLV)**:
  - Segment-based CLV prediction
  - Marketing budget allocation by segment
  - High-value customer targeting

#### 3. **Operational Excellence**
- **Staffing Optimization**:
  - Demand-based scheduling using hourly forecasts
  - Cross-training recommendations (peak vs. off-peak skills)
  - Break scheduling based on customer flow
- **Inventory Management**:
  - Demand forecasting per product per hour
  - Stock level optimization to reduce waste
  - Supplier coordination based on demand patterns
  - Just-in-time inventory modeling
- **Supply Chain Analytics**:
  - If supplier data available: lead time optimization
  - Batch size optimization
  - Supplier performance metrics
- **Capacity Planning**:
  - Queue modeling during peak hours
  - Wait time predictions
  - Equipment/staffing capacity analysis

#### 4. **Advanced Predictive Modeling**
- **Deep Learning Forecasting**:
  - LSTM/GRU models for time-series prediction
  - Multi-step ahead forecasting (1 week - 1 month)
  - Uncertainty quantification (prediction intervals)
- **Ensemble Methods**:
  - Combine multiple models (Random Forest, XGBoost, ARIMA, Prophet)
  - Weighted ensemble for improved accuracy
  - Model stacking for meta-predictions
- **Causal Analysis**:
  - If external data available: weather impact on sales
  - Holiday/event impact quantification
  - Promotion effectiveness measurement
- **Real-time Forecasting**:
  - Streaming data integration
  - Online learning models
  - Live demand predictions

#### 5. **Customer Experience & Behavior**
- **Purchase Pattern Mining**:
  - Market basket analysis (association rules)
  - Sequential pattern discovery
  - Product recommendation engine
- **Customer Sentiment** (if reviews available):
  - Sentiment analysis on feedback
  - Topic modeling of complaints/compliments
  - NLP-based customer satisfaction scoring
- **Demographic Analysis** (if customer data available):
  - Age/gender-based preferences
  - Loyalty program effectiveness by segment
  - Location-based analysis (if multi-location)
- **Experience Mapping**:
  - Customer journey visualization by segment
  - Pain point identification
  - Improvement prioritization

#### 6. **Competitive & Market Intelligence**
- **Benchmark Analysis**:
  - Industry standard comparison (if data available)
  - Competitive positioning
  - Best practice implementation
- **Market Trend Analysis**:
  - Emerging product preferences
  - Seasonal demand evolution
  - Category growth analysis
- **Expansion Opportunities**:
  - New product viability assessment
  - New location feasibility
  - Menu optimization recommendations

#### 7. **Risk Management & Quality Control**
- **Quality Metrics** (if quality data available):
  - Product defect rates
  - Customer satisfaction correlation
  - Returns/complaints analysis
- **Sales Variance Analysis**:
  - Explain unexpected deviations
  - Identify root causes of anomalies
  - Mitigation strategies for risks
- **Cash Flow Forecasting**:
  - Revenue predictability
  - Seasonal cash flow planning
  - Working capital optimization

#### 8. **Sustainability & Compliance**
- **Waste Analysis**:
  - If inventory tracking available: waste by product
  - Spoilage rate analysis
  - Sustainability reporting
- **Compliance Metrics**:
  - If POS system data available: discrepancy tracking
  - Register accuracy analysis
  - Audit trail analytics

#### 9. **Advanced Visualization & Dashboards**
- **Interactive Dashboards**:
  - Streamlit app with real-time updates
  - Dash for stakeholder reporting
  - Power BI integration
  - Drill-down capabilities by date, product, segment
- **Real-time Monitoring**:
  - KPI alerts and notifications
  - Anomaly notifications
  - Performance tracking dashboard
- **Custom Reports**:
  - Executive summaries
  - Weekly/monthly performance reports
  - Custom ad-hoc analysis delivery

#### 10. **Hypothesis Testing & Experimentation**
- **Statistical Testing**:
  - T-tests for weekend vs. weekday differences
  - ANOVA for product performance comparison
  - Chi-square tests for categorical relationships
  - Correlation significance testing
- **A/B Testing Framework**:
  - Menu change impact measurement
  - Pricing experiment design
  - Promotion effectiveness testing
- **Experimental Design**:
  - Sample size calculation
  - Power analysis
  - Multiple testing correction

---

## Tech Stack

- **Python 3.13+**
- **Data & Analysis**:
  - pandas – Data manipulation and aggregation
  - numpy – Numerical operations
  - scipy – Statistical analysis
- **Machine Learning**:
  - scikit-learn – Regression (RandomForest), clustering (KMeans), preprocessing (StandardScaler)
- **Visualization**:
  - matplotlib – Static plots
  - seaborn – Statistical visualizations
- **Optional/Future**:
  - kagglehub – Data sourcing
  - statsmodels – Advanced time-series (ARIMA, seasonal decomposition)
  - prophet – Forecasting
  - streamlit – Interactive dashboards
  - plotly – Interactive visualizations

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

### Quick Start - Run All Analyses

#### Option 1: Foundational EDA Only
```bash
# Modular, organized output
python eda_coffee.py

# Or interactive with plots
python coffee_sales_data_analysis.py

# Or comprehensive with recommendations
python basic_analysis.py
```

#### Option 2: Foundational + Phase 1 Advanced Analytics
```bash
# Complete Phase 1: Forecasting + Segmentation + Anomalies
python run_phase1.py
```

#### Option 3: Jupyter Notebook (Kaggle Format)
```bash
jupyter notebook eda-analysis-on-daily-coffee-transaction.ipynb
```

### Output Locations
All analysis results automatically saved to `/outputs/`:
- CSV data files with metrics and results
- PNG visualization dashboards (300 DPI, high quality)
- Ready for presentations and reports

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

## Key Insights from Analysis ✨

### Business Performance
- ✅ **Peak Hours Identified**: Precise hourly demand patterns for staffing optimization
- ✅ **Weekday/Weekend Patterns**: Clear differences in customer behavior by day type
- ✅ **Product Performance**: Top performers drive majority of revenue (Pareto principle)
- ✅ **Price Variations**: Multiple price points per product suggest variants (sizes, add-ons)
- ✅ **Payment Diversity**: Multiple payment methods used across customer base

### Phase 1 Advanced Analytics
- ✅ **Revenue Forecasting**: Machine learning model predicts next 7 days with quantified accuracy
- ✅ **Customer Segments**: 4 distinct behavioral segments identified with unique characteristics
- ✅ **Anomaly Detection**: Unusual revenue days and transactions flagged for investigation
- ✅ **Customer Journey**: Clear morning → afternoon → evening progression pattern
- ✅ **Segment Profitability**: Different segments show different revenue contribution patterns

### Strategic Recommendations (from analyses)
1. **Staffing**: Optimize for peak hours using hourly demand forecasts
2. **Inventory**: Stock high-volume items heavily; rotate slow movers to peak hours
3. **Marketing**: Target off-peak hours with promotions to balance daily demand
4. **Pricing**: Consider dynamic pricing for peak vs. off-peak periods
5. **Operations**: Extend hours if off-peak segment justifies expansion; reduce if not profitable
6. **Product Mix**: Prioritize high-revenue items; develop niche products for evening segment
7. **Risk Management**: Investigate anomalies for root causes (events, system errors, unusual behavior)

---

## Next Steps & Roadmap

### Immediate (Phase 1 - Complete ✅)
- [x] Foundational exploratory data analysis
- [x] Time-series forecasting with ML
- [x] Customer segmentation and journey mapping
- [x] Anomaly detection and outlier identification
- [x] Strategic recommendations generation

### Short-term (Phase 2 - Planned)
- [ ] Interactive Streamlit/Dash dashboard for stakeholder monitoring
- [ ] Hypothesis testing for statistical significance of patterns
- [ ] A/B testing framework for menu/pricing experiments
- [ ] If cost data available: profit margin analysis
- [ ] If customer IDs available: RFM analysis and churn prediction

### Medium-term (Phase 3 - Expansion)
- [ ] Deep learning models (LSTM) for improved forecasting
- [ ] Causal analysis with external variables (weather, events, holidays)
- [ ] Product recommendation engine
- [ ] Dynamic pricing optimization
- [ ] Advanced inventory management modeling

### Long-term (Phase 4+ - Strategic)
- [ ] Multi-location analysis and comparison
- [ ] Customer sentiment analysis (if reviews available)
- [ ] Market expansion feasibility studies
- [ ] Competitive benchmarking
- [ ] Sustainability and efficiency metrics

---

## Project Progression & Development Timeline

### Version 0.1 - Foundational EDA ✅
- Initial data exploration and cleaning
- Descriptive statistics and summary metrics
- Time-series and temporal pattern analysis
- Product performance evaluation
- Statistical analysis and correlation studies
- **Scripts**: `eda_coffee.py`, `coffee_sales_data_analysis.py`

### Version 0.2 - Enhanced Foundational Analysis ✅
- Comprehensive business overview with strategic recommendations
- Detailed temporal analysis (hourly, daily, weekly, time-of-day)
- Product-level detailed metrics and performance ranking
- Weekend vs. weekday comparative analysis
- Business optimization recommendations
- 4-panel comprehensive dashboard
- **Scripts**: `basic_analysis.py`

### Version 1.0 - Advanced Analytics (Phase 1) ✅
- **Forecasting Module**: Random Forest revenue prediction (7-day forecast)
- **Segmentation Module**: K-means clustering into 4 customer segments
- **Anomaly Detection Module**: Z-score and IQR-based outlier identification
- Feature importance analysis for revenue drivers
- Customer journey mapping
- 13 output files (CSV + PNG)
- **Scripts**: `run_phase1.py`

### Version 1.1 - Jupyter Notebook ✅
- Interactive Kaggle-format notebook
- Markdown documentation
- Reproducible analysis cells
- **Scripts**: `eda-analysis-on-daily-coffee-transaction.ipynb`

### Version 2.0 - Phase 2 (Planned)
- Interactive dashboards (Streamlit/Dash)
- Statistical hypothesis testing
- A/B testing framework
- Cost analysis (if data available)
- Customer ID analysis (if available)

### Version 3.0 - Phase 3+ (Future)
- Deep learning forecasting models
- Causal analysis with external variables
- Recommendation systems
- Dynamic pricing optimization
- Advanced inventory management

---

## Data Enhancement Questions

To unlock additional analysis capabilities and insights, consider providing:

### Customer & Transaction Data
- **Customer IDs**: Enable RFM analysis, cohort analysis, churn prediction, loyalty tracking
- **Repeat Purchase History**: Build customer lifetime value models and personalization
- **Discount/Promo Codes**: Measure promotion effectiveness and ROI
- **Customer Demographics**: Age, location for segmentation and targeting

### Business Operations Data
- **Cost Data**: Product cost, labor, rent for margin analysis and profitability
- **Inventory Levels**: Stock quantities for demand-supply optimization
- **Staff Schedules**: Actual schedules vs. demand for efficiency analysis
- **Supplier Information**: Lead times, batch sizes, costs

### External Context
- **Weather Data**: Correlation with daily/hourly demand patterns
- **Holiday/Event Calendar**: Impact on sales, special event days
- **Competitor Data**: Market share, pricing benchmarking
- **Macro Indicators**: Economic data, seasonal trends

### Quality & Feedback
- **Customer Reviews/Ratings**: Sentiment analysis, quality metrics
- **Complaints/Returns Log**: Quality issues, product problems
- **Social Media Data**: Brand sentiment, product mentions
- **Survey Responses**: Customer satisfaction metrics

### Operational Metrics
- **POS System Logs**: Detailed transaction times, order accuracy
- **Queue Data**: Wait times, service levels
- **Waste/Spoilage**: Product loss tracking
- **Equipment Maintenance**: Downtime impact

---

## How to Use This Repository

### For Quick Insights
1. Run: `python basic_analysis.py`
2. Check `outputs/comprehensive_analysis.png` for 4-panel dashboard
3. Read printed strategic recommendations

### For Detailed Exploration
1. Run: `python coffee_sales_data_analysis.py`
2. Interact with plot windows for deeper investigation
3. Review console output for detailed metrics

### For Advanced Predictive Insights
1. Run: `python run_phase1.py`
2. Review 9 output files (CSV + PNG)
3. Focus on forecast results and segment profitability

### For Interactive Development
1. Open Jupyter: `jupyter notebook eda-analysis-on-daily-coffee-transaction.ipynb`
2. Execute cells to explore step-by-step
3. Modify code for custom analysis

### For Integration & Reporting
- All outputs saved to `/outputs/` with high-resolution PNG (300 DPI)
- CSV files ready for Excel, PowerBI, or Tableau
- Structured data for automated report generation

---

## Project Statistics

- **Total Python Scripts**: 4 (varying complexity levels)
- **Jupyter Notebooks**: 1 (Kaggle-compatible)
- **Analysis Phases**: 3 (Foundational, Phase 1A-C)
- **Output Files**: 30+ (CSV + PNG)
- **ML Models Implemented**: 2 (RandomForest, KMeans)
- **Visualizations**: 13+ dashboards and plots
- **Key Metrics Tracked**: 50+
- **Lines of Code**: 1500+

---

## Summary

This comprehensive coffee shop analysis project provides:

✅ **Actionable Insights** from transaction-level data  
✅ **Predictive Capabilities** for revenue forecasting  
✅ **Customer Intelligence** through behavioral segmentation  
✅ **Risk Management** via anomaly detection  
✅ **Strategic Recommendations** for operations & marketing  
✅ **Scalable Framework** for continuous improvement  
✅ **Production-Ready Output** (CSV + visualizations)  
✅ **Multiple Analysis Levels** (beginner to advanced)  

**Use Case**: Perfect for coffee shop owners, managers, analysts, and data scientists seeking data-driven decisions on pricing, staffing, inventory, and marketing.

**Next Steps**: Implement Phase 2 for interactive dashboards and automation, or use Phase 1 insights to drive immediate operational improvements.

---

## License & Attribution

This analysis is part of a comprehensive data analysis learning project on exploratory data analysis (EDA) and advanced analytics for small business intelligence and optimization.

**Project Start Date**: November 2025  
**Current Version**: 1.1 (Phase 1 Complete)  
**Author**: Rendy Bagoez

For questions, improvements, or contributions, please refer to the project repository.
