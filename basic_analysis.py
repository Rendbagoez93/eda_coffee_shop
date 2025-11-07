
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("COMPREHENSIVE COFFEE SALES DATA ANALYSIS")
print("="*60)

# Load the data
df = pd.read_csv("data/Coffe_sales.csv")

print(f"DATASET OVERVIEW")
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"FIRST 5 ROWS:")
print(df.head())

print(f"DATASET INFO:")
df.info()

print(f"DATA QUALITY CHECK:")
print("Missing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values)

print(f"BASIC STATISTICS:")
print(df.describe())

print("DATA PREPROCESSING")
print("="*40)

# Standardize column names
df.columns = [col.strip().lower().replace(' ', '_') 
              for col in df.columns]
print("Standardized Column Names:", df.columns.tolist())

# Data type conversions with error handling
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.time
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    print("Date/Time conversion successful")
except Exception as e:
    print(f"Error converting time data: {e}")

# Convert categorical columns
cat_cols = ['cash_type', 'coffee_name', 'time_of_day', 'weekday', 'month_name']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
print("Categorical conversion complete")

# Ensure numerical columns
df['money'] = pd.to_numeric(df['money'], errors='coerce')
df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')

# Add derived columns
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month  
df['year'] = df['date'].dt.year
df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday', 'Sat', 'Sun'])

print("Data preprocessing complete!")
print(f"Final dataset shape: {df.shape}")

print("\n" + "="*60)
print("BUSINESS PERFORMANCE OVERVIEW")
print("="*60)

# Core Business Metrics
total_revenue = df['money'].sum()
avg_transaction = df['money'].mean()
total_transactions = len(df)
date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
operating_hours = f"{df['hour_of_day'].min()}:00 to {df['hour_of_day'].max()}:00"

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Transaction: ${avg_transaction:.2f}")
print(f"Total Transactions: {total_transactions:,}")
print(f"Business Period: {date_range}")
print(f"Operating Hours: {operating_hours}")
print(f"Payment Methods: {df['cash_type'].nunique()} method(s)")

# Top performing products
print(f"\nTOP PRODUCTS BY SALES VOLUME:")
coffee_counts = df['coffee_name'].value_counts()
for i, (coffee, count) in enumerate(coffee_counts.head().items(), 1):
    print(f"  {i}. {coffee}: {count:,} orders")

print(f"\nTOP PRODUCTS BY REVENUE:")
coffee_revenue = df.groupby('coffee_name')['money'].sum().sort_values(ascending=False)
for i, (coffee, revenue) in enumerate(coffee_revenue.head().items(), 1):
    print(f"  {i}. {coffee}: ${revenue:,.2f}")

# Payment analysis
print(f"PAYMENT METHOD BREAKDOWN:")
payment_dist = df['cash_type'].value_counts()
for method, count in payment_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  {method}: {count:,} transactions ({percentage:.1f}%)")

# Time-based summary
print(f"WEEKLY PERFORMANCE SUMMARY:")
weekday_stats = df.groupby('weekday')['money'].agg(['count', 'sum']).round(2)
weekday_stats.columns = ['transactions', 'revenue']
weekday_stats = weekday_stats.sort_values('revenue', ascending=False)
for day, stats in weekday_stats.head().iterrows():
    print(f"  {day}: {stats['transactions']} orders, ${stats['revenue']:,.2f}")

print("\n" + "="*60)
print("TEMPORAL ANALYSIS & PATTERNS")
print("="*60)

# Hourly patterns
print("HOURLY PERFORMANCE:")
hourly_stats = df.groupby('hour_of_day').agg({
    'money': ['count', 'sum', 'mean']
}).round(2)
hourly_stats.columns = ['transactions', 'total_revenue', 'avg_transaction']
peak_hour = hourly_stats['total_revenue'].idxmax()
print(f"Peak Revenue Hour: {peak_hour}:00 (${hourly_stats.loc[peak_hour, 'total_revenue']:,.2f})")

# Time period analysis
print("TIME PERIOD BREAKDOWN:")
timeofday_stats = df.groupby('time_of_day').agg({
    'money': ['count', 'sum', 'mean']
}).round(2)
timeofday_stats.columns = ['transactions', 'total_revenue', 'avg_transaction']
for period, stats in timeofday_stats.iterrows():
    print(f"  {period}: {stats['transactions']} transactions, ${stats['total_revenue']:,.2f}")

# Weekend vs Weekday comparison
print("WEEKEND vs WEEKDAY ANALYSIS:")
weekend_comparison = df.groupby('is_weekend').agg({
    'money': ['count', 'sum', 'mean']
}).round(2)
weekend_comparison.columns = ['transactions', 'total_revenue', 'avg_transaction']
weekend_comparison.index = ['Weekdays', 'Weekends']
print(weekend_comparison)
    
print("\n" + "="*60)
print("COMPREHENSIVE VISUALIZATIONS")
print("="*60)

# Create comprehensive dashboard
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Hourly Revenue Pattern
hourly_revenue = df.groupby('hour_of_day')['money'].sum()
axes[0, 0].bar(hourly_revenue.index, hourly_revenue.values, color='steelblue', alpha=0.8)
axes[0, 0].set_title('Revenue by Hour of Day', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Total Revenue ($)')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Weekday Revenue Pattern
weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekday_revenue = df.groupby('weekday')['money'].sum().reindex(weekday_order)
colors = ['lightcoral' if day in ['Sat', 'Sun'] else 'skyblue' for day in weekday_order]
axes[0, 1].bar(weekday_revenue.index, weekday_revenue.values, color=colors, alpha=0.8)
axes[0, 1].set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Total Revenue ($)')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Top Products Revenue
top_products = df.groupby('coffee_name')['money'].sum().sort_values(ascending=False).head(8)
axes[1, 0].barh(range(len(top_products)), top_products.values, color='forestgreen', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_products)))
axes[1, 0].set_yticklabels(top_products.index)
axes[1, 0].set_title('Top 8 Products by Revenue', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Total Revenue ($)')
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Transaction Amount Distribution  
axes[1, 1].hist(df['money'], bins=25, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Transaction Amount ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("STATISTICAL INSIGHTS:")
print(f"Price Range: ${df['money'].min():.2f} - ${df['money'].max():.2f}")
print(f"Most Common Prices:")
price_counts = df['money'].value_counts().head()
for price, count in price_counts.items():
    print(f"  ${price}: {count} transactions")

print(f"\nSkewness & Distribution:")
print(f"  Money - Skewness: {df['money'].skew():.3f}")
print(f"  Hour - Skewness: {df['hour_of_day'].skew():.3f}")

print("DETAILED PRODUCT ANALYSIS:")
product_analysis = df.groupby('coffee_name').agg({
    'money': ['count', 'sum', 'mean'],
    'hour_of_day': 'mean'
}).round(2)
product_analysis.columns = ['transactions', 'total_revenue', 'avg_price', 'avg_hour_sold']
product_analysis = product_analysis.sort_values('total_revenue', ascending=False)
print(product_analysis.head(10))

print("\n" + "="*60)
print("KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Generate key insights
hourly_revenue = df.groupby('hour_of_day')['money'].sum()
weekday_revenue = df.groupby('weekday')['money'].sum()
peak_hour = hourly_revenue.idxmax()
peak_day = weekday_revenue.idxmax()
top_product = df.groupby('coffee_name')['money'].sum().idxmax()

print("PEAK PERFORMANCE METRICS:")
print(f"  • Peak Revenue Hour: {peak_hour}:00 (${hourly_revenue.loc[peak_hour]:,.2f})")
print(f"  • Best Performing Day: {peak_day} (${weekday_revenue.loc[peak_day]:,.2f})")
print(f"  • Top Revenue Product: {top_product}")
print(f"  • Highest Transaction: ${df['money'].max():.2f}")
print(f"  • Business operates {df['hour_of_day'].max() - df['hour_of_day'].min() + 1} hours/day")

print("CUSTOMER BEHAVIOR PATTERNS:")
morning_sales = df[df['time_of_day'] == 'Morning']['money'].count()
afternoon_sales = df[df['time_of_day'] == 'Afternoon']['money'].count() 
night_sales = df[df['time_of_day'] == 'Night']['money'].count()

print(f"  • Morning Rush: {morning_sales:,} transactions ({morning_sales/len(df)*100:.1f}%)")
print(f"  • Afternoon Peak: {afternoon_sales:,} transactions ({afternoon_sales/len(df)*100:.1f}%)")
print(f"  • Evening Sales: {night_sales:,} transactions ({night_sales/len(df)*100:.1f}%)")

# Weekend vs Weekday insights
weekday_trans = len(df[~df['is_weekend']])
weekend_trans = len(df[df['is_weekend']])
print(f"  • Weekday Dominance: {weekday_trans:,} transactions ({weekday_trans/len(df)*100:.1f}%)")
print(f"  • Weekend Activity: {weekend_trans:,} transactions ({weekend_trans/len(df)*100:.1f}%)")

print("STRATEGIC RECOMMENDATIONS:")
print("  1. STAFFING: Optimize for 10-11 AM peak hours")
print("  2. INVENTORY: Stock premium items (Latte, Cappuccino) during weekdays")  
print("  3. MARKETING: Develop weekend promotions to boost Sat/Sun sales")
print("  4. PRICING: Consider dynamic pricing for peak vs off-peak hours")
print("  5. OPERATIONS: Extend hours if night sales justify costs")

print("SAVING ANALYSIS RESULTS...")

# Save detailed analysis to CSV files
try:
    # Hourly analysis
    hourly_analysis = df.groupby('hour_of_day').agg({
        'money': ['count', 'sum', 'mean'],
        'coffee_name': lambda x: x.value_counts().index[0]  # Most popular item
    }).round(2)
    hourly_analysis.columns = ['transactions', 'revenue', 'avg_transaction', 'top_product']
    hourly_analysis.to_csv('outputs/hourly_detailed_analysis.csv')

    # Daily analysis
    daily_analysis = df.groupby('weekday').agg({
        'money': ['count', 'sum', 'mean'],
        'coffee_name': lambda x: x.value_counts().index[0]
    }).round(2)
    daily_analysis.columns = ['transactions', 'revenue', 'avg_transaction', 'top_product'] 
    daily_analysis.to_csv('outputs/daily_detailed_analysis.csv')

    # Product performance analysis
    product_performance = df.groupby('coffee_name').agg({
        'money': ['count', 'sum', 'mean'],
        'hour_of_day': ['mean', 'min', 'max'],
        'weekday': lambda x: x.value_counts().index[0]  # Most popular day
    }).round(2)
    product_performance.columns = ['orders', 'total_revenue', 'avg_price', 'avg_hour', 'first_sale_hour', 'last_sale_hour', 'peak_day']
    product_performance = product_performance.sort_values('total_revenue', ascending=False)
    product_performance.to_csv('outputs/product_performance_detailed.csv')

    print("Analysis files saved successfully:")
    print("  • outputs/comprehensive_analysis.png")
    print("  • outputs/hourly_detailed_analysis.csv")
    print("  • outputs/daily_detailed_analysis.csv") 
    print("  • outputs/product_performance_detailed.csv")

except Exception as e:
    print(f"Error saving files: {e}")

print(f"\nANALYSIS COMPLETE!")
print(f"Dataset processed: {len(df):,} transactions")
print(f"Total business value: ${df['money'].sum():,.2f}")
print(f"Ready for business optimization!")
print("="*60)
