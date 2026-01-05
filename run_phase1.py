"""
Phase 1 Advanced Analytics: Time Series Forecasting & Customer Segmentation
Imports data and preprocessing from basic_analysis.py
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_processed_data():
    """Load and preprocess data using basic_analysis logic"""
    from basic_analysis import pd, os
    
    # Load the data (same as basic_analysis.py)
    df = pd.read_csv("data/Coffe_sales.csv")
    
    # Apply same preprocessing as basic_analysis.py
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.time
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    except Exception as e:
        print(f"Error converting time data: {e}")
    
    # Convert categorical columns
    cat_cols = ['cash_type', 'coffee_name', 'time_of_day', 'weekday', 'month_name']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Ensure numerical columns
    df['money'] = pd.to_numeric(df['money'], errors='coerce')
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')
    
    # Add derived columns
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month  
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday', 'Sat', 'Sun'])
    
    return df

def time_series_forecasting(df):
    """Advanced Time Series Analysis and Forecasting"""
    print("\n" + "="*60)
    print("🔮 PHASE 1A: TIME SERIES FORECASTING & DEMAND PREDICTION")
    print("="*60)
    
    # Create daily aggregated data for forecasting
    daily_data = df.groupby('date').agg({
        'money': 'sum',
        'coffee_name': 'count'
    }).rename(columns={'coffee_name': 'transactions'}).reset_index()
    
    # Add time features for modeling
    daily_data['day_of_year'] = daily_data['date'].dt.dayofyear
    daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Create rolling averages for trend analysis
    daily_data['revenue_7day_avg'] = daily_data['money'].rolling(window=7).mean()
    daily_data['revenue_30day_avg'] = daily_data['money'].rolling(window=30).mean()
    daily_data['revenue_volatility'] = daily_data['money'].rolling(window=7).std()
    
    # Prepare features for forecasting model
    features = ['day_of_year', 'day_of_week', 'month', 'is_weekend']
    X = daily_data[features].fillna(daily_data[features].mean())
    y = daily_data['money'].fillna(daily_data['money'].mean())
    
    # Split data for training/testing (80/20 split)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train forecasting model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"📈 REVENUE FORECASTING PERFORMANCE:")
    print(f"  • Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  • Root Mean Square Error (RMSE): ${rmse:.2f}")
    print(f"  • Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
    
    print(f"\n🔍 FEATURE IMPORTANCE FOR REVENUE PREDICTION:")
    feature_importance = list(zip(features, rf_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importance:
        print(f"  • {feature.replace('_', ' ').title()}: {importance:.3f}")
    
    # Seasonal analysis
    daily_data['trend'] = daily_data['revenue_30day_avg']
    daily_data['seasonal'] = daily_data['money'] - daily_data['trend']
    
    print(f"\n📊 SEASONAL DECOMPOSITION ANALYSIS:")
    print(f"  • Average Daily Revenue: ${daily_data['money'].mean():.2f}")
    print(f"  • Revenue Volatility (std): ${daily_data['money'].std():.2f}")
    print(f"  • Coefficient of Variation: {(daily_data['money'].std()/daily_data['money'].mean())*100:.1f}%")
    
    # Future predictions (next 7 days)
    print(f"\n🔮 NEXT 7 DAYS REVENUE FORECAST:")
    last_date = daily_data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
    
    future_features = []
    for date in future_dates:
        future_features.append([
            date.dayofyear,
            date.dayofweek, 
            date.month,
            1 if date.dayofweek in [5, 6] else 0
        ])
    
    future_X = pd.DataFrame(future_features, columns=features)
    future_predictions = rf_model.predict(future_X)
    
    for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
        day_type = "Weekend" if date.dayofweek in [5, 6] else "Weekday"
        print(f"  • {date.strftime('%Y-%m-%d')} ({date.strftime('%A')}): ${pred:.2f} ({day_type})")
    
    # Create forecasting visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.plot(daily_data['date'][:split_point], daily_data['money'][:split_point], 
             label='Training Data', alpha=0.7, color='blue')
    plt.plot(daily_data['date'][split_point:], y_test, 
             label='Actual', color='green', linewidth=2)
    plt.plot(daily_data['date'][split_point:], y_pred, 
             label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title('Revenue Forecasting: Actual vs Predicted', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Revenue ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Trend Analysis
    plt.subplot(2, 2, 2)
    plt.plot(daily_data['date'], daily_data['money'], alpha=0.3, label='Daily Revenue')
    plt.plot(daily_data['date'], daily_data['revenue_7day_avg'], 
             color='orange', linewidth=2, label='7-Day Average')
    plt.plot(daily_data['date'], daily_data['revenue_30day_avg'], 
             color='red', linewidth=2, label='30-Day Average')
    plt.title('Revenue Trend Analysis', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 3: Seasonal Pattern
    plt.subplot(2, 2, 3)
    seasonal_by_day = daily_data.groupby('day_of_week')['money'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.bar(days, seasonal_by_day.values, color='skyblue', alpha=0.8)
    plt.title('Average Revenue by Day of Week', fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Revenue ($)')
    
    # Plot 4: Volatility Analysis
    plt.subplot(2, 2, 4)
    plt.plot(daily_data['date'], daily_data['revenue_volatility'], 
             color='purple', alpha=0.7)
    plt.title('Revenue Volatility (7-Day Rolling Std)', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Volatility ($)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/phase1_forecasting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save forecasting results
    forecasting_results = daily_data.copy()
    forecasting_results['predicted_revenue'] = np.nan
    forecasting_results.loc[split_point:, 'predicted_revenue'] = y_pred
    forecasting_results.to_csv('outputs/phase1_forecasting_results.csv', index=False)
    
    return daily_data, rf_model

def customer_segmentation_analysis(df):
    """Advanced Customer Behavior Segmentation"""
    print("\n" + "="*60)
    print("🎯 PHASE 1B: CUSTOMER SEGMENTATION & BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Create customer behavior features by hour and product preference
    hourly_behavior = df.groupby(['hour_of_day', 'coffee_name']).size().unstack(fill_value=0)
    hourly_spending = df.groupby('hour_of_day')['money'].agg(['mean', 'sum', 'count', 'std'])
    
    # Time-based customer segments
    time_segments = []
    for hour in sorted(df['hour_of_day'].unique()):
        hour_data = df[df['hour_of_day'] == hour]
        segment = {
            'hour': hour,
            'avg_transaction': hour_data['money'].mean(),
            'total_customers': len(hour_data),
            'total_revenue': hour_data['money'].sum(),
            'top_product': hour_data['coffee_name'].mode().iloc[0] if len(hour_data) > 0 else 'N/A',
            'spending_variance': hour_data['money'].var(),
            'customer_type': hour_data['time_of_day'].mode().iloc[0] if len(hour_data) > 0 else 'N/A'
        }
        time_segments.append(segment)
    
    time_segments_df = pd.DataFrame(time_segments)
    
    # K-means clustering for hour-based segments
    scaler = StandardScaler()
    features_for_clustering = ['avg_transaction', 'total_customers', 'spending_variance']
    
    # Handle missing values
    clustering_data = time_segments_df[features_for_clustering].fillna(0)
    scaled_features = scaler.fit_transform(clustering_data)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    k_range = range(2, 8)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    # Use 4 clusters based on business intuition (morning, afternoon, evening, night)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    time_segments_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    print(f"🏷️ CUSTOMER TIME-BASED SEGMENTS ({optimal_k} clusters identified):")
    
    cluster_names = {0: "High-Value Low-Volume", 1: "Peak Rush Hours", 2: "Steady Customers", 3: "Off-Peak Hours"}
    
    for cluster in sorted(time_segments_df['cluster'].unique()):
        cluster_data = time_segments_df[time_segments_df['cluster'] == cluster]
        cluster_name = cluster_names.get(cluster, f"Segment {cluster + 1}")
        
        print(f"\n  📊 {cluster_name}:")
        print(f"    • Hours: {sorted(cluster_data['hour'].tolist())}")
        print(f"    • Avg Transaction: ${cluster_data['avg_transaction'].mean():.2f}")
        print(f"    • Avg Customers/Hour: {cluster_data['total_customers'].mean():.0f}")
        print(f"    • Total Revenue: ${cluster_data['total_revenue'].sum():,.2f}")
        
        # Most popular products in this segment
        segment_hours = cluster_data['hour'].tolist()
        segment_products = df[df['hour_of_day'].isin(segment_hours)]['coffee_name'].value_counts().head(3)
        print(f"    • Top Products: {', '.join(segment_products.index.tolist())}")
        
        behavior_type = "Premium" if cluster_data['avg_transaction'].mean() > df['money'].mean() else "Volume"
        print(f"    • Customer Type: {behavior_type}")
    
    # Product affinity analysis
    print(f"\n🛍️ PRODUCT AFFINITY BY TIME SEGMENTS:")
    
    for cluster in sorted(time_segments_df['cluster'].unique()):
        cluster_hours = time_segments_df[time_segments_df['cluster'] == cluster]['hour'].tolist()
        cluster_name = cluster_names.get(cluster, f"Segment {cluster + 1}")
        
        cluster_sales = df[df['hour_of_day'].isin(cluster_hours)]
        product_performance = cluster_sales.groupby('coffee_name')['money'].agg(['count', 'sum', 'mean']).round(2)
        product_performance.columns = ['orders', 'revenue', 'avg_price']
        product_performance = product_performance.sort_values('revenue', ascending=False)
        
        print(f"\n  {cluster_name} - Top 3 Products:")
        for i, (product, stats) in enumerate(product_performance.head(3).iterrows(), 1):
            print(f"    {i}. {product}: {stats['orders']} orders, ${stats['revenue']:.2f} revenue")
    
    # Customer journey analysis
    print(f"\n🚶 CUSTOMER JOURNEY ANALYSIS:")
    
    # Analyze typical customer patterns throughout the day
    hourly_flow = df.groupby('hour_of_day').agg({
        'money': ['count', 'sum', 'mean'],
        'coffee_name': lambda x: x.mode().iloc[0]
    }).round(2)
    hourly_flow.columns = ['customers', 'revenue', 'avg_spend', 'popular_product']
    
    # Identify journey stages
    total_daily_customers = hourly_flow['customers'].sum()
    
    print(f"  🌅 Morning Rush (6-11 AM):")
    morning_hours = hourly_flow.loc[6:11]
    print(f"    • Customers: {morning_hours['customers'].sum()} ({morning_hours['customers'].sum()/total_daily_customers*100:.1f}%)")
    print(f"    • Revenue: ${morning_hours['revenue'].sum():,.2f}")
    print(f"    • Peak Hour: {morning_hours['customers'].idxmax()}:00 ({morning_hours['customers'].max()} customers)")
    
    print(f"  ☀️ Afternoon Peak (12-5 PM):")
    afternoon_hours = hourly_flow.loc[12:17]
    print(f"    • Customers: {afternoon_hours['customers'].sum()} ({afternoon_hours['customers'].sum()/total_daily_customers*100:.1f}%)")
    print(f"    • Revenue: ${afternoon_hours['revenue'].sum():,.2f}")
    print(f"    • Peak Hour: {afternoon_hours['customers'].idxmax()}:00 ({afternoon_hours['customers'].max()} customers)")
    
    print(f"  🌙 Evening Wind-down (6-10 PM):")
    evening_hours = hourly_flow.loc[18:22]
    print(f"    • Customers: {evening_hours['customers'].sum()} ({evening_hours['customers'].sum()/total_daily_customers*100:.1f}%)")
    print(f"    • Revenue: ${evening_hours['revenue'].sum():,.2f}")
    print(f"    • Peak Hour: {evening_hours['customers'].idxmax()}:00 ({evening_hours['customers'].max()} customers)")
    
    # Create segmentation visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Customer segments by hour
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    for cluster in sorted(time_segments_df['cluster'].unique()):
        cluster_data = time_segments_df[time_segments_df['cluster'] == cluster]
        plt.scatter(cluster_data['hour'], cluster_data['total_customers'], 
                   c=colors[cluster], label=cluster_names.get(cluster, f'Segment {cluster+1}'),
                   s=100, alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Customers')
    plt.title('Customer Segments by Hour', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Transaction value by segment
    plt.subplot(2, 3, 2)
    for cluster in sorted(time_segments_df['cluster'].unique()):
        cluster_data = time_segments_df[time_segments_df['cluster'] == cluster]
        plt.scatter(cluster_data['hour'], cluster_data['avg_transaction'], 
                   c=colors[cluster], label=cluster_names.get(cluster, f'Segment {cluster+1}'),
                   s=100, alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Transaction ($)')
    plt.title('Transaction Value by Segment', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Hourly customer flow
    plt.subplot(2, 3, 3)
    hourly_customers = df.groupby('hour_of_day').size()
    plt.plot(hourly_customers.index, hourly_customers.values, 
             marker='o', linewidth=2, markersize=6, color='purple')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Customers')
    plt.title('Daily Customer Flow Pattern', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Revenue distribution by segment
    plt.subplot(2, 3, 4)
    segment_revenue = time_segments_df.groupby('cluster')['total_revenue'].sum()
    plt.pie(segment_revenue.values, 
            labels=[cluster_names.get(i, f'Segment {i+1}') for i in segment_revenue.index],
            autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Revenue Distribution by Segment', fontweight='bold')
    
    # Plot 5: Customer behavior heatmap
    plt.subplot(2, 3, 5)
    pivot_data = df.groupby(['hour_of_day', 'weekday']).size().unstack(fill_value=0)
    sns.heatmap(pivot_data.T, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Customer Count'})
    plt.title('Customer Activity Heatmap', fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    # Plot 6: Cluster characteristics
    plt.subplot(2, 3, 6)
    cluster_summary = time_segments_df.groupby('cluster')[['avg_transaction', 'total_customers']].mean()
    x = range(len(cluster_summary))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], cluster_summary['avg_transaction'], width, 
            label='Avg Transaction ($)', alpha=0.8)
    plt.bar([i + width/2 for i in x], cluster_summary['total_customers']/10, width, 
            label='Customers (÷10)', alpha=0.8)
    
    plt.xlabel('Cluster')
    plt.ylabel('Value')
    plt.title('Cluster Characteristics Comparison', fontweight='bold')
    plt.xticks(x, [cluster_names.get(i, f'Segment {i+1}') for i in cluster_summary.index], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/phase1_customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save segmentation results
    time_segments_df.to_csv('outputs/phase1_customer_segments.csv', index=False)
    hourly_flow.to_csv('outputs/phase1_customer_journey.csv', index=True)
    
    return time_segments_df, hourly_flow

def anomaly_detection(df):
    """Detect anomalous patterns in sales data"""
    print("\n" + "="*60)
    print("🚨 PHASE 1C: ANOMALY DETECTION & OUTLIER ANALYSIS")
    print("="*60)
    
    # Daily revenue anomaly detection
    daily_revenue = df.groupby('date')['money'].sum().reset_index()
    daily_revenue['day_of_week'] = daily_revenue['date'].dt.dayofweek
    daily_revenue['is_weekend'] = daily_revenue['day_of_week'].isin([5, 6])
    
    # Statistical anomaly detection using Z-score
    revenue_stats = daily_revenue.groupby('is_weekend')['money'].agg(['mean', 'std'])
    daily_revenue['z_score'] = daily_revenue.apply(
        lambda row: (row['money'] - revenue_stats.loc[row['is_weekend'], 'mean']) / 
                    revenue_stats.loc[row['is_weekend'], 'std'], axis=1
    )
    
    # Identify unusual days (z-score > 2 or < -2)
    anomalous_days = daily_revenue[abs(daily_revenue['z_score']) > 2]
    
    print(f"📊 STATISTICAL ANOMALY DETECTION RESULTS:")
    print(f"  • Total Days Analyzed: {len(daily_revenue)}")
    print(f"  • Anomalous Days Detected: {len(anomalous_days)} ({len(anomalous_days)/len(daily_revenue)*100:.1f}%)")
    
    if len(anomalous_days) > 0:
        print(f"\n  🔍 UNUSUAL REVENUE DAYS:")
        for _, day in anomalous_days.sort_values('z_score', key=abs, ascending=False).head(10).iterrows():
            anomaly_type = "Exceptionally High" if day['z_score'] > 0 else "Unusually Low"
            day_type = "Weekend" if day['is_weekend'] else "Weekday"
            print(f"    • {day['date'].date()} ({day_type}): ${day['money']:.2f}")
            print(f"      {anomaly_type} revenue (z-score: {day['z_score']:.2f})")
    
    # Transaction-level anomaly detection
    print(f"\n💰 TRANSACTION-LEVEL ANOMALY ANALYSIS:")
    
    # IQR method for transaction amounts
    Q1 = df['money'].quantile(0.25)
    Q3 = df['money'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    transaction_outliers = df[(df['money'] < lower_bound) | (df['money'] > upper_bound)]
    
    print(f"  • Transaction Range: ${df['money'].min():.2f} - ${df['money'].max():.2f}")
    print(f"  • Normal Range (IQR): ${lower_bound:.2f} - ${upper_bound:.2f}")
    print(f"  • Outlier Transactions: {len(transaction_outliers)} ({len(transaction_outliers)/len(df)*100:.1f}%)")
    
    if len(transaction_outliers) > 0:
        high_value = transaction_outliers[transaction_outliers['money'] > upper_bound]
        low_value = transaction_outliers[transaction_outliers['money'] < lower_bound]
        
        print(f"    - High-value outliers: {len(high_value)} transactions")
        print(f"    - Low-value outliers: {len(low_value)} transactions")
        
        if len(high_value) > 0:
            top_high_value = high_value.nlargest(5, 'money')
            print(f"    - Top unusual high transactions:")
            for _, txn in top_high_value.iterrows():
                print(f"      ${txn['money']:.2f} - {txn['coffee_name']} at {txn['hour_of_day']}:00")
    
    # Hourly pattern anomalies
    print(f"\n🕐 HOURLY PATTERN ANOMALIES:")
    
    hourly_avg = df.groupby(['hour_of_day', 'weekday'])['money'].mean().reset_index()
    hourly_overall = df.groupby('hour_of_day')['money'].mean()
    
    unusual_hour_day_combos = []
    for hour in df['hour_of_day'].unique():
        hour_data = hourly_avg[hourly_avg['hour_of_day'] == hour]
        hour_mean = hourly_overall[hour]
        hour_std = df[df['hour_of_day'] == hour]['money'].std()
        
        for _, combo in hour_data.iterrows():
            z_score = (combo['money'] - hour_mean) / hour_std if hour_std > 0 else 0
            if abs(z_score) > 1.5:  # Less strict threshold for hourly data
                unusual_hour_day_combos.append({
                    'hour': combo['hour_of_day'],
                    'weekday': combo['weekday'],
                    'avg_revenue': combo['money'],
                    'z_score': z_score
                })
    
    if unusual_hour_day_combos:
        unusual_df = pd.DataFrame(unusual_hour_day_combos)
        unusual_df = unusual_df.sort_values('z_score', key=abs, ascending=False).head(10)
        
        print(f"  • Unusual Hour-Day Combinations Found: {len(unusual_df)}")
        for _, combo in unusual_df.iterrows():
            pattern_type = "Higher than expected" if combo['z_score'] > 0 else "Lower than expected"
            print(f"    • {combo['weekday']} at {combo['hour']}:00: {pattern_type}")
            print(f"      Avg transaction: ${combo['avg_revenue']:.2f} (z-score: {combo['z_score']:.2f})")
    
    # Create anomaly visualization
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Daily revenue with anomalies highlighted
    plt.subplot(2, 2, 1)
    plt.plot(daily_revenue['date'], daily_revenue['money'], alpha=0.7, color='blue', linewidth=1)
    if len(anomalous_days) > 0:
        plt.scatter(anomalous_days['date'], anomalous_days['money'], 
                   color='red', s=50, alpha=0.8, label=f'Anomalies ({len(anomalous_days)})')
    plt.title('Daily Revenue with Anomalies', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Revenue ($)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Z-score distribution
    plt.subplot(2, 2, 2)
    plt.hist(daily_revenue['z_score'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(-2, color='red', linestyle='--', label='Anomaly Threshold')
    plt.axvline(2, color='red', linestyle='--')
    plt.title('Z-Score Distribution (Daily Revenue)', fontweight='bold')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Transaction amount outliers
    plt.subplot(2, 2, 3)
    plt.boxplot(df['money'])
    plt.title('Transaction Amount Distribution\n(Outliers as Points)', fontweight='bold')
    plt.ylabel('Transaction Amount ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Anomaly timeline
    plt.subplot(2, 2, 4)
    daily_revenue['anomaly'] = abs(daily_revenue['z_score']) > 2
    monthly_anomalies = daily_revenue.groupby(daily_revenue['date'].dt.to_period('M'))['anomaly'].sum()
    monthly_anomalies.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Anomalies by Month', fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Number of Anomalous Days')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/phase1_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save anomaly results
    daily_revenue.to_csv('outputs/phase1_daily_anomalies.csv', index=False)
    if len(transaction_outliers) > 0:
        transaction_outliers.to_csv('outputs/phase1_transaction_outliers.csv', index=False)
    
    return daily_revenue, transaction_outliers

def main():
    """Run Phase 1 Advanced Analytics"""
    print("🚀 STARTING PHASE 1: ADVANCED ANALYTICS")
    print("Loading and preprocessing data...")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load processed data
    df = load_processed_data()
    print(f"✅ Data loaded successfully: {len(df):,} transactions")
    
    # Run Phase 1 analyses
    daily_data, forecast_model = time_series_forecasting(df)
    segments_df, journey_df = customer_segmentation_analysis(df)
    anomaly_results, outliers = anomaly_detection(df)
    
    # Generate Phase 1 summary report
    print("\n" + "="*60)
    print("📋 PHASE 1 SUMMARY REPORT")
    print("="*60)
    
    print(f"✅ COMPLETED ANALYSES:")
    print(f"  • Time Series Forecasting & Demand Prediction")
    print(f"  • Customer Segmentation (4 segments identified)")
    print(f"  • Anomaly Detection & Outlier Analysis")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"  • phase1_forecasting_results.csv")
    print(f"  • phase1_customer_segments.csv")
    print(f"  • phase1_customer_journey.csv")
    print(f"  • phase1_daily_anomalies.csv")
    print(f"  • phase1_forecasting_analysis.png")
    print(f"  • phase1_customer_segmentation.png")
    print(f"  • phase1_anomaly_detection.png")
    
    print(f"\n🎯 KEY INSIGHTS:")
    peak_revenue_day = daily_data.loc[daily_data['money'].idxmax(), 'date'].strftime('%Y-%m-%d')
    avg_daily_revenue = daily_data['money'].mean()
    
    print(f"  • Best Revenue Day: {peak_revenue_day} (${daily_data['money'].max():,.2f})")
    print(f"  • Average Daily Revenue: ${avg_daily_revenue:,.2f}")
    print(f"  • Revenue Forecasting Accuracy: Ready for 7-day predictions")
    print(f"  • Customer Segments: 4 distinct behavioral patterns identified")
    print(f"  • Anomalies Detected: {len(anomaly_results[abs(anomaly_results['z_score']) > 2])} unusual days")
    
    print(f"\n🚀 READY FOR PHASE 2: Business Optimization Analytics")
    print("="*60)

if __name__ == "__main__":
    main()