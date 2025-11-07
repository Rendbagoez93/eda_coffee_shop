
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