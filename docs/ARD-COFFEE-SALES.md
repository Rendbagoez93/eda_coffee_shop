# Coffee Sales Analysis – Analysis Requirements Document (ARD)

## 1. Background

The Coffee Sales dataset contains transaction-level data enriched with time, calendar, product, and payment attributes. This analysis aims to transform raw sales data into actionable business insights that support decision-making for revenue growth, operational efficiency, and product optimization.

This ARD defines the scope, objectives, key business questions, analyses, and expected outcomes of the Coffee Sales Analysis initiative.

---

## 2. Business Objectives

The primary objectives of this analysis are:

1. **Revenue Optimization**
   Identify opportunities to increase total revenue through better pricing, promotions, and product focus.

2. **Operational Efficiency**
   Understand demand patterns to optimize staffing, operating hours, and inventory planning.

3. **Product & Marketing Strategy**
   Evaluate product performance and customer preferences to inform menu design and targeted marketing.

---

## 3. Dataset Overview (Based on Kaggle Public Dataset by Minahil Fatima)

### Available Columns

* **hour_of_day** – Hour when transaction occurred
* **cash_type** – Payment method (cash / non-cash)
* **money** – Transaction amount
* **coffee_name** – Product name
* **Time_of_Day** – Categorized time block (e.g., Morning, Afternoon, Evening)
* **Weekday** – Day name (Monday–Sunday)
* **Weekdaysort** – Numeric weekday order
* **Month_name** – Month name
* **Monthsort** – Numeric month order
* **Date** – Transaction date
* **Time** – Transaction time

---

## 4. Key Business Questions & Analysis Scope

### 4.1 Sales Performance Analysis

**Business Questions**

* How much total revenue is generated?
* Which coffee products contribute the most and least revenue?
* How does revenue change over time?

**Analysis Scope**

* Total revenue and transaction count
* Revenue by coffee type
* Revenue by day, weekday, and month
* Average transaction value

**Expected Insights**

* Identification of top-performing and underperforming products
* Revenue concentration across products and periods
* High-value vs low-value sales periods

**Business Impact**

* Product prioritization
* Pricing and promotion adjustments
* Performance benchmarking

---

### 4.2 Time-Based Demand Analysis

**Business Questions**

* What are the peak and off-peak hours?
* How does demand differ by time of day?
* Are there weekday vs weekend differences?

**Analysis Scope**

* Sales and revenue by hour_of_day
* Sales by Time_of_Day
* Sales by Weekday
* Hour vs Weekday demand heatmap

**Expected Insights**

* Clear identification of rush hours
* Low-demand time slots
* Day-specific demand patterns

**Business Impact**

* Staff scheduling optimization
* Operating hour refinement
* Time-based promotional strategies

---

### 4.3 Product Preference Analysis

**Business Questions**

* Which coffee products are most popular?
* Do product preferences vary by time or day?
* Are certain products time-specific?

**Analysis Scope**

* Product sales volume and revenue
* Product performance by Time_of_Day
* Product performance by hour and weekday

**Expected Insights**

* Product–time alignment (e.g., morning vs afternoon drinks)
* Niche products with consistent demand
* Products with limited selling windows

**Business Impact**

* Menu optimization
* Time-based promotions
* Inventory planning by time slot

---

### 4.4 Payment Behavior Analysis

**Business Questions**

* What payment methods do customers prefer?
* Does payment type influence spending value?
* Does payment behavior vary by time?

**Analysis Scope**

* Transaction count by cash_type
* Revenue contribution by payment method
* Payment method usage by hour and weekday

**Expected Insights**

* Cash vs cashless adoption trends
* Differences in average spend by payment type
* Time-based payment preferences

**Business Impact**

* POS system optimization
* Cash handling efficiency
* Payment-based marketing initiatives

---

### 4.5 Seasonality & Calendar Analysis

**Business Questions**

* Do sales fluctuate by month?
* Are there identifiable seasonal patterns?

**Analysis Scope**

* Monthly revenue trends
* Month-over-month growth analysis
* Comparison using Monthsort

**Expected Insights**

* High and low sales seasons
* Predictable sales cycles
* Planning signals for inventory and staffing

**Business Impact**

* Inventory forecasting
* Campaign and budget planning
* Revenue forecasting

---

## 5. Cross-Dimensional & Advanced Analysis (Optional)

**Examples**

* Best-selling coffee by time of day
* Peak hours by weekday
* Payment preference by time segment

**Expected Value**

* Deeper behavioral insights
* Executive-ready narratives combining multiple dimensions

---

## 6. Deliverables

* Analytical summary report
* KPI definitions and metrics
* Visual dashboards (charts, tables, heatmaps)
* Actionable business recommendations

---

## 7. Success Metrics

* Clear identification of peak demand periods
* Actionable product performance insights
* Practical recommendations adopted by business stakeholders
* Improved decision-making clarity for operations and marketing

---

## 8. Assumptions & Constraints

* Dataset represents completed transactions only
* No customer-level identifiers are available
* External factors (weather, holidays) are not explicitly included

---

## 9. Conclusion

This ARD outlines a structured and business-focused approach to analyzing Coffee Sales data. The resulting insights are expected to directly support revenue growth, operational efficiency, and informed product and marketing strategies.
