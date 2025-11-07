
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/Coffe_sales.csv")

print("Dataset Shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Info summary
print("Dataset Info:")
df.info()

# Check missing values
print("Missing Values per Column:")
print(df.isnull().sum())

# Quick statistical summary
print("Descriptive Statistics (numerical columns):")
print(df.describe())

# Standarize column names
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print("Standardized Column Names:", df.columns.tolist())

print("Missing Values per Column After Standardization:")
print(df.isnull().sum())

# Converting Data Types - Add Error Handling
# Convert date & time
try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Check if time column needs conversion
    if df['time'].dtype == 'object':
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.time
    else:
        df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.time
except Exception as e:
    print(f"Error converting time data: {e}")

# Combine into datetime if needed
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

# Convert categorical data
cat_cols = ['cash_type', 'coffee_name', 'time_of_day', 'weekday', 'month_name']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Convert numerical
df['money'] = pd.to_numeric(df['money'], errors='coerce')
df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')


print("=" * 40)

# Revenue & Transaction Overview
total_revenue = df['money'].sum()
avg_transaction = df['money'].mean()
total_transactions = len(df)

print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Average Transaction Value: {avg_transaction:,.2f}")
print(f"Total Transactions: {total_transactions}")

print("Top 5 Coffee Types:")
print(df['coffee_name'].value_counts().head())

print("Payment Method Distribution:")
print(df['cash_type'].value_counts())

print("Sales by Weekday:")
print(df['weekday'].value_counts())

# Add day, month, year for time-based grouping
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Create weekend flag
df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])

# Example: revenue per hour
revenue_per_hour = df.groupby('hour_of_day')['money'].sum().reset_index()
revenue_per_day = df.groupby('date')['money'].sum().reset_index()

# Simple check on sales pattern
sns.barplot(x='hour_of_day', y='money', data=df, estimator='sum')
plt.title('Revenue by Hour of Day')
plt.show()

print ("Unique Weekday values in data:")
print(df['weekday'].unique())
print("\nWeekday value counts:")
print(df['weekday'].value_counts())

# Check for missing data in the plotting columns
print(f"\nMissing values - weekday: {df['weekday'].isnull().sum()}")
print(f"Missing values - money: {df['money'].isnull().sum()}")

# Plot with error handling and debugging
plt.figure(figsize=(10, 6))
try:
    # First, let's see what weekdays actually exist in your data
    actual_weekdays = df['weekday'].unique()
    print(f"Actual weekdays in data: {actual_weekdays}")
    
    # Plot without forcing order first
    sns.barplot(data=df, x='weekday', y='money', estimator='sum')
    plt.title('Revenue by Weekday')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error creating weekday plot: {e}")
    
print ("=" * 40)

# Statiistical Analysis
df[['money', 'hour_of_day']].describe()

print("Skewness:\n", df[['money', 'hour_of_day']].skew())
print("Kurtosis:\n", df[['money', 'hour_of_day']].kurt())

# Distribution Plots
plt.figure(figsize=(7,5))
sns.histplot(df['money'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Money Spent')
plt.ylabel('Frequency')
plt.show()

# Boxplot for outliers
sns.boxplot(x='money', data=df)
plt.title('Outlier Detection in Transaction Amount')
plt.show()

# Correlation Analysis
corr = df[['hour_of_day', 'money']].corr()
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

corr_cols = ['money', 'hour_of_day', 'weekdaysort', 'monthsort']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='vlag')
plt.title('Correlation Across Time Dimensions')
plt.show()

# Group-Based Statistical Summaries
print("Statistical Summary by Cash Type:")
print(df.groupby('cash_type')['money'].describe())

# Average by weekday
print ("Statistical Summary by Weekday:")
weekday_stats = df.groupby('weekday')['money'].agg(['mean','median','std','count'])
print(weekday_stats)

# Average by time of day
print ("Statistical Summary by Time of Day:")
timeofday_stats = df.groupby('time_of_day')['money'].agg(['mean','median','std','count'])
print(timeofday_stats)


# Product Performance
coffee_stats = df.groupby('coffee_name')['money'].agg(['count','mean','sum']).sort_values('sum', ascending=False)
print(coffee_stats.head(10))

# Hourly & Daily Trends (Aggregated)
hourly_sales = df.groupby('hour_of_day')['money'].sum().reset_index()
sns.lineplot(x='hour_of_day', y='money', data=hourly_sales)
plt.title('Total Revenue by Hour of Day')
plt.show()

# Daily revenue by weekday (use a fixed order and handle missing days)
weekdays_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# Ensure weekday column uses consistent ordering (safe coercion)
try:
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekdays_order, ordered=True)
except Exception:
    # if conversion fails, continue — groupby + reindex below will still handle ordering
    pass

daily_sales = (
    df.groupby('weekday', sort=False)['money']
      .sum()
      .reindex(weekdays_order)
      .fillna(0)
      .reset_index()
      .rename(columns={'money': 'money'})
)

plt.figure(figsize=(10, 6))
sns.barplot(x='weekday', y='money', data=daily_sales, order=weekdays_order)
plt.title('Total Revenue by Weekday')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
