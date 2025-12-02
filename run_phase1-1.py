"""
ADVANCED ANALYTICS: DEMAND PREDICTION & BEHAVIORAL ANALYSIS
===========================================================

This module provides sophisticated analytical capabilities for coffee shop data:
1. Demand Forecasting using Multiple Models
2. Customer Behavioral Analysis & Segmentation
3. Advanced Statistical Analysis & Patterns
4. Predictive Models for Business Optimization
5. Interactive Visualizations and Insights

Author: Rendy Herdianto
Date: December 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import timedelta
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import chi2_contingency


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Create outputs directory structure
os.makedirs('outputs/advanced', exist_ok=True)
os.makedirs('outputs/predictions', exist_ok=True)
os.makedirs('outputs/behavioral', exist_ok=True)

class CoffeeShopAdvancedAnalytics:
    """
    Advanced Analytics Engine for Coffee Shop Business Intelligence
    """
    
    def __init__(self, data_file='data/Coffe_sales.csv'):
        """Initialize the analytics engine with data preprocessing"""
        print("🚀 Initializing Advanced Coffee Shop Analytics Engine...")
        self.load_and_preprocess_data(data_file)
        self.feature_engineering()
        print(f"✅ Data loaded: {len(self.df):,} transactions over {self.df['date'].nunique()} days")
        
    def load_and_preprocess_data(self, data_file):
        """Load and preprocess the coffee shop data"""
        # Load raw data
        self.df = pd.read_csv(data_file)
        
        # Standardize column names
        self.df.columns = [col.strip().lower().replace(' ', '_') 
                          for col in self.df.columns]
        
        # Data type conversions
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['datetime'] = pd.to_datetime(self.df['date'].astype(str) + ' ' + self.df['time'], errors='coerce')
        
        # Extract time components
        self.df['time_parsed'] = pd.to_datetime(self.df['time'], errors='coerce').dt.time
        
        # Ensure numerical columns
        self.df['money'] = pd.to_numeric(self.df['money'], errors='coerce')
        self.df['hour_of_day'] = pd.to_numeric(self.df['hour_of_day'], errors='coerce')
        
        # Create categorical columns
        cat_cols = ['cash_type', 'coffee_name', 'time_of_day', 'weekday', 'month_name']
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
    
    def feature_engineering(self):
        """Create advanced features for modeling"""
        print("🔧 Engineering advanced features...")
        
        # Temporal features
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['day_of_month'] = self.df['datetime'].dt.day
        self.df['week_of_year'] = self.df['datetime'].dt.isocalendar().week
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        self.df['is_month_start'] = self.df['day_of_month'] <= 5
        self.df['is_month_end'] = self.df['day_of_month'] >= 25
        
        # Business logic features
        self.df['is_peak_hour'] = self.df['hour_of_day'].between(8, 10) | self.df['hour_of_day'].between(14, 16)
        self.df['is_lunch_time'] = self.df['hour_of_day'].between(11, 14)
        self.df['is_evening'] = self.df['hour_of_day'] >= 17
        
        # Price categories
        self.df['price_tier'] = pd.cut(self.df['money'], 
                                      bins=[0, 30, 35, 40, float('inf')], 
                                      labels=['Budget', 'Standard', 'Premium', 'Luxury'])
        
        # Customer behavior proxies
        self.df['transaction_size'] = self.df['money'] / self.df['money'].median()
        
    def demand_forecasting_suite(self):
        """Comprehensive demand forecasting with multiple models"""
        print("\n" + "="*70)
        print("📈 DEMAND FORECASTING & PREDICTION MODELS")
        print("="*70)
        
        # Prepare time series data
        daily_data = self.df.groupby('date').agg({
            'money': ['sum', 'count', 'mean'],
            'hour_of_day': 'mean',
            'is_weekend': 'first'
        }).round(2)
        
        daily_data.columns = ['daily_revenue', 'transaction_count', 'avg_transaction', 
                             'avg_hour', 'is_weekend']
        daily_data = daily_data.reset_index()
        
        # Feature engineering for forecasting
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['day_of_month'] = daily_data['date'].dt.day
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['days_since_start'] = (daily_data['date'] - daily_data['date'].min()).dt.days
        
        # Add lag features
        for lag in [1, 3, 7]:
            daily_data[f'revenue_lag_{lag}'] = daily_data['daily_revenue'].shift(lag)
            daily_data[f'count_lag_{lag}'] = daily_data['transaction_count'].shift(lag)
        
        # Moving averages
        daily_data['revenue_ma_3'] = daily_data['daily_revenue'].rolling(window=3).mean()
        daily_data['revenue_ma_7'] = daily_data['daily_revenue'].rolling(window=7).mean()
        
        # Drop rows with NaN values from lag features
        modeling_data = daily_data.dropna()
        
        print(f"📊 Forecasting dataset: {len(modeling_data)} days of data")
        
        # Define features and target
        feature_cols = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'days_since_start',
                       'revenue_lag_1', 'revenue_lag_3', 'revenue_lag_7',
                       'count_lag_1', 'count_lag_3', 'count_lag_7',
                       'revenue_ma_3', 'revenue_ma_7', 'avg_transaction']
        
        X = modeling_data[feature_cols]
        y_revenue = modeling_data['daily_revenue']
        y_count = modeling_data['transaction_count']
        
        # Split data for training/testing
        split_point = int(len(modeling_data) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_rev_train, y_rev_test = y_revenue[:split_point], y_revenue[split_point:]
        y_count_train, y_count_test = y_count[:split_point], y_count[split_point:]
        
        # Initialize models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # Train and evaluate models for revenue prediction
        revenue_results = {}
        count_results = {}
        
        print("\n🎯 REVENUE PREDICTION MODELS:")
        print("-" * 50)
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_rev_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Evaluate
            train_mae = mean_absolute_error(y_rev_train, train_pred)
            test_mae = mean_absolute_error(y_rev_test, test_pred)
            train_r2 = r2_score(y_rev_train, train_pred)
            test_r2 = r2_score(y_rev_test, test_pred)
            
            revenue_results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': test_pred
            }
            
            print(f"📈 {name}:")
            print(f"   Training MAE: ${train_mae:.2f} | R²: {train_r2:.3f}")
            print(f"   Testing MAE:  ${test_mae:.2f} | R²: {test_r2:.3f}")
        
        print("\n🎯 TRANSACTION COUNT PREDICTION MODELS:")
        print("-" * 50)
        
        for name, model in models.items():
            # Create new model instance for count prediction
            if name == 'Random Forest':
                count_model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif name == 'Gradient Boosting':
                count_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                count_model = LinearRegression()
            
            # Train model
            count_model.fit(X_train, y_count_train)
            
            # Predictions
            train_pred = count_model.predict(X_train)
            test_pred = count_model.predict(X_test)
            
            # Evaluate
            train_mae = mean_absolute_error(y_count_train, train_pred)
            test_mae = mean_absolute_error(y_count_test, test_pred)
            train_r2 = r2_score(y_count_train, train_pred)
            test_r2 = r2_score(y_count_test, test_pred)
            
            count_results[name] = {
                'model': count_model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': test_pred
            }
            
            print(f"📊 {name}:")
            print(f"   Training MAE: {train_mae:.1f} transactions | R²: {train_r2:.3f}")
            print(f"   Testing MAE:  {test_mae:.1f} transactions | R²: {test_r2:.3f}")
        
        # Select best models
        best_revenue_model = min(revenue_results.items(), key=lambda x: x[1]['test_mae'])
        best_count_model = min(count_results.items(), key=lambda x: x[1]['test_mae'])
        
        print(f"\n🏆 BEST MODELS:")
        print(f"   Revenue Prediction: {best_revenue_model[0]} (MAE: ${best_revenue_model[1]['test_mae']:.2f})")
        print(f"   Count Prediction: {best_count_model[0]} (MAE: {best_count_model[1]['test_mae']:.1f})")
        
        # Generate future predictions
        self.generate_future_forecasts(modeling_data, best_revenue_model[1]['model'], 
                                     best_count_model[1]['model'], feature_cols)
        
        # Create prediction visualization
        self.create_prediction_visualizations(modeling_data, revenue_results, count_results, 
                                            X_test, y_rev_test, y_count_test)
        
        return revenue_results, count_results
    
    def generate_future_forecasts(self, historical_data, revenue_model, count_model, feature_cols):
        """Generate forecasts for the next 30 days"""
        print("\n🔮 GENERATING 30-DAY FORECASTS...")
        
        # Create future date range
        last_date = historical_data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        # Create future features
        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['day_of_month'] = future_df['date'].dt.day
        future_df['month'] = future_df['date'].dt.month
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6])
        future_df['days_since_start'] = (future_df['date'] - historical_data['date'].min()).dt.days
        
        # Use last known values for lag features (simplified approach)
        last_revenue = historical_data['daily_revenue'].iloc[-7:].values
        last_counts = historical_data['transaction_count'].iloc[-7:].values
        
        # Initialize forecasts
        revenue_forecasts = []
        count_forecasts = []
        
        for i, row in future_df.iterrows():
            # Create feature vector
            features = [
                row['day_of_week'], row['day_of_month'], row['month'],
                row['is_weekend'], row['days_since_start']
            ]
            
            # Add lag features (use recent historical data)
            if i == 0:
                features.extend([last_revenue[-1], last_revenue[-3], last_revenue[-7]])
                features.extend([last_counts[-1], last_counts[-3], last_counts[-7]])
                features.extend([np.mean(last_revenue[-3:]), np.mean(last_revenue[-7:])])
                features.append(last_revenue[-1] / last_counts[-1] if last_counts[-1] > 0 else 35)
            else:
                # Use previous predictions for lag features
                lag_1_rev = revenue_forecasts[-1] if i >= 1 else last_revenue[-1]
                lag_3_rev = revenue_forecasts[-3] if i >= 3 else last_revenue[-(3-i)] if (3-i) < len(last_revenue) else last_revenue[-1]
                lag_7_rev = revenue_forecasts[-7] if i >= 7 else last_revenue[-(7-i)] if (7-i) < len(last_revenue) else last_revenue[-1]
                
                lag_1_count = count_forecasts[-1] if i >= 1 else last_counts[-1]
                lag_3_count = count_forecasts[-3] if i >= 3 else last_counts[-(3-i)] if (3-i) < len(last_counts) else last_counts[-1]
                lag_7_count = count_forecasts[-7] if i >= 7 else last_counts[-(7-i)] if (7-i) < len(last_counts) else last_counts[-1]
                
                features.extend([lag_1_rev, lag_3_rev, lag_7_rev])
                features.extend([lag_1_count, lag_3_count, lag_7_count])
                
                # Moving averages
                recent_rev = revenue_forecasts[-3:] if i >= 3 else [last_revenue[-1]] * (3-i) + revenue_forecasts
                recent_rev_7 = revenue_forecasts[-7:] if i >= 7 else list(last_revenue[-(7-i):]) + revenue_forecasts if i < 7 else revenue_forecasts[-7:]
                
                ma_3 = np.mean(recent_rev[-3:])
                ma_7 = np.mean(recent_rev_7[-7:])
                avg_transaction = lag_1_rev / lag_1_count if lag_1_count > 0 else 35
                
                features.extend([ma_3, ma_7, avg_transaction])
            
            # Make predictions
            X_pred = np.array(features).reshape(1, -1)
            rev_pred = revenue_model.predict(X_pred)[0]
            count_pred = count_model.predict(X_pred)[0]
            
            revenue_forecasts.append(max(0, rev_pred))  # Ensure positive values
            count_forecasts.append(max(1, count_pred))  # Ensure at least 1 transaction
        
        # Create forecast dataframe
        forecast_df = future_df.copy()
        forecast_df['predicted_revenue'] = revenue_forecasts
        forecast_df['predicted_transactions'] = count_forecasts
        forecast_df['predicted_avg_transaction'] = forecast_df['predicted_revenue'] / forecast_df['predicted_transactions']
        
        # Add confidence intervals (simple approach using historical volatility)
        revenue_std = historical_data['daily_revenue'].std()
        count_std = historical_data['transaction_count'].std()
        
        forecast_df['revenue_lower_ci'] = forecast_df['predicted_revenue'] - 1.96 * revenue_std
        forecast_df['revenue_upper_ci'] = forecast_df['predicted_revenue'] + 1.96 * revenue_std
        forecast_df['count_lower_ci'] = forecast_df['predicted_transactions'] - 1.96 * count_std
        forecast_df['count_upper_ci'] = forecast_df['predicted_transactions'] + 1.96 * count_std
        
        # Save forecasts
        forecast_df.to_csv('outputs/predictions/30_day_forecasts.csv', index=False)
        
        # Display summary
        total_predicted_revenue = forecast_df['predicted_revenue'].sum()
        avg_daily_revenue = forecast_df['predicted_revenue'].mean()
        total_predicted_transactions = forecast_df['predicted_transactions'].sum()
        
        print(f"📊 30-Day Forecast Summary:")
        print(f"   Total Predicted Revenue: ${total_predicted_revenue:,.2f}")
        print(f"   Average Daily Revenue: ${avg_daily_revenue:.2f}")
        print(f"   Total Predicted Transactions: {total_predicted_transactions:.0f}")
        print(f"   Average Transactions/Day: {forecast_df['predicted_transactions'].mean():.1f}")
        
        # Weekend vs Weekday predictions
        weekend_forecast = forecast_df[forecast_df['is_weekend']]
        weekday_forecast = forecast_df[~forecast_df['is_weekend']]
        
        print(f"\n📅 Weekend vs Weekday Forecasts:")
        print(f"   Weekday Average: ${weekday_forecast['predicted_revenue'].mean():.2f}/day")
        print(f"   Weekend Average: ${weekend_forecast['predicted_revenue'].mean():.2f}/day")
        
        return forecast_df
    
    def create_prediction_visualizations(self, historical_data, revenue_results, count_results, 
                                       X_test, y_rev_test, y_count_test):
        """Create comprehensive prediction visualizations"""
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Prediction Accuracy', 'Transaction Count Prediction',
                          'Model Performance Comparison', 'Prediction vs Actual'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Revenue prediction accuracy
        best_revenue_model = min(revenue_results.items(), key=lambda x: x[1]['test_mae'])
        predictions = best_revenue_model[1]['predictions']
        
        fig.add_trace(
            go.Scatter(x=list(range(len(y_rev_test))), y=y_rev_test.values, 
                      mode='lines+markers', name='Actual Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(predictions))), y=predictions, 
                      mode='lines+markers', name='Predicted Revenue', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. Transaction count prediction
        best_count_model = min(count_results.items(), key=lambda x: x[1]['test_mae'])
        count_predictions = best_count_model[1]['predictions']
        
        fig.add_trace(
            go.Scatter(x=list(range(len(y_count_test))), y=y_count_test.values, 
                      mode='lines+markers', name='Actual Transactions', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(count_predictions))), y=count_predictions, 
                      mode='lines+markers', name='Predicted Transactions', line=dict(color='orange', dash='dash')),
            row=1, col=2
        )
        
        # 3. Model performance comparison
        model_names = list(revenue_results.keys())
        rev_mae_scores = [revenue_results[name]['test_mae'] for name in model_names]
        count_mae_scores = [count_results[name]['test_mae'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=rev_mae_scores, name='Revenue MAE', marker_color='lightblue'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=model_names, y=count_mae_scores, name='Count MAE', marker_color='lightcoral', yaxis='y2'),
            row=2, col=1
        )
        
        # 4. Scatter plot: Prediction vs Actual
        fig.add_trace(
            go.Scatter(x=y_rev_test.values, y=predictions, mode='markers', 
                      name='Revenue Predictions', marker=dict(color='purple', size=8)),
            row=2, col=2
        )
        
        # Add perfect prediction line
        min_val = min(y_rev_test.min(), predictions.min())
        max_val = max(y_rev_test.max(), predictions.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dot')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Advanced Demand Prediction Analysis",
            height=800,
            showlegend=True
        )
        
        # Save plot
        fig.write_html('outputs/predictions/prediction_analysis.html')
        
        print("💾 Prediction visualizations saved to outputs/predictions/")
    
    def customer_behavioral_analysis(self):
        """Comprehensive customer behavior analysis and segmentation"""
        print("\n" + "="*70)
        print("👥 CUSTOMER BEHAVIORAL ANALYSIS & SEGMENTATION")
        print("="*70)
        
        # Customer behavior features aggregation (using date+hour as proxy for customer sessions)
        customer_sessions = self.df.groupby(['date', 'hour_of_day']).agg({
            'money': ['sum', 'count', 'mean'],
            'coffee_name': lambda x: x.nunique(),
            'cash_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'is_weekend': 'first',
            'is_peak_hour': 'first'
        }).round(2)
        
        customer_sessions.columns = ['total_spent', 'items_purchased', 'avg_item_price', 
                                   'unique_products', 'preferred_payment', 'is_weekend', 'is_peak_hour']
        customer_sessions = customer_sessions.reset_index()
        
        print(f"📊 Analyzed {len(customer_sessions):,} customer sessions")
        
        # Behavioral features for clustering
        behavioral_features = customer_sessions[['total_spent', 'items_purchased', 
                                               'avg_item_price', 'unique_products']].copy()
        
        # Standardize features for clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(behavioral_features)
        
        # K-means clustering for customer segmentation
        print("\n🎯 CUSTOMER SEGMENTATION ANALYSIS:")
        print("-" * 40)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 8)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
        
        # Use 4 clusters as optimal (business logic)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        customer_sessions['segment'] = kmeans.fit_predict(scaled_features)
        
        # Analyze segments
        segment_analysis = customer_sessions.groupby('segment').agg({
            'total_spent': ['count', 'mean', 'median', 'std'],
            'items_purchased': ['mean', 'median'],
            'avg_item_price': ['mean', 'median'],
            'unique_products': ['mean', 'median'],
            'is_weekend': 'mean',
            'is_peak_hour': 'mean'
        }).round(2)
        
        # Define segment names based on characteristics
        segment_names = {
            0: 'Budget Conscious',
            1: 'Premium Customers',
            2: 'Bulk Buyers',
            3: 'Casual Visitors'
        }
        
        # Find actual segment characteristics and assign names
        segment_summary = {}
        for seg in range(optimal_k):
            seg_data = customer_sessions[customer_sessions['segment'] == seg]
            avg_spent = seg_data['total_spent'].mean()
            avg_items = seg_data['items_purchased'].mean()
            avg_price = seg_data['avg_item_price'].mean()
            
            if avg_spent > customer_sessions['total_spent'].mean() and avg_price > customer_sessions['avg_item_price'].mean():
                name = 'Premium Customers'
            elif avg_items > customer_sessions['items_purchased'].mean():
                name = 'Bulk Buyers'
            elif avg_spent < customer_sessions['total_spent'].mean():
                name = 'Budget Conscious'
            else:
                name = 'Regular Customers'
            
            segment_summary[seg] = {
                'name': name,
                'size': len(seg_data),
                'avg_spent': avg_spent,
                'avg_items': avg_items,
                'avg_price': avg_price,
                'weekend_preference': seg_data['is_weekend'].mean(),
                'peak_hour_preference': seg_data['is_peak_hour'].mean()
            }
        
        print("📈 Customer Segments Identified:")
        for seg, info in segment_summary.items():
            print(f"\n🏷️  Segment {seg + 1}: {info['name']}")
            print(f"    Size: {info['size']:,} sessions ({info['size']/len(customer_sessions)*100:.1f}%)")
            print(f"    Avg Spending: ${info['avg_spent']:.2f}")
            print(f"    Avg Items: {info['avg_items']:.1f}")
            print(f"    Avg Item Price: ${info['avg_price']:.2f}")
            print(f"    Weekend Preference: {info['weekend_preference']*100:.1f}%")
            print(f"    Peak Hour Visits: {info['peak_hour_preference']*100:.1f}%")
        
        # Save segmentation results
        customer_sessions.to_csv('outputs/behavioral/customer_segments.csv', index=False)
        
        # Purchase pattern analysis
        self.analyze_purchase_patterns()
        
        # Payment behavior analysis
        self.analyze_payment_behaviors()
        
        # Create behavioral visualizations
        self.create_behavioral_visualizations(customer_sessions, segment_summary)
        
        return customer_sessions, segment_summary
    
    def analyze_purchase_patterns(self):
        """Analyze detailed purchase patterns and preferences"""
        print("\n🛒 PURCHASE PATTERN ANALYSIS:")
        print("-" * 40)
        
        # Product affinity analysis
        product_combinations = self.df.groupby(['date', 'hour_of_day'])['coffee_name'].apply(list)
        
        # Single vs Multiple item sessions
        session_sizes = self.df.groupby(['date', 'hour_of_day']).size()
        single_item_sessions = (session_sizes == 1).sum()
        multi_item_sessions = (session_sizes > 1).sum()
        
        print(f"📊 Session Analysis:")
        print(f"   Single-item sessions: {single_item_sessions:,} ({single_item_sessions/(single_item_sessions+multi_item_sessions)*100:.1f}%)")
        print(f"   Multi-item sessions: {multi_item_sessions:,} ({multi_item_sessions/(single_item_sessions+multi_item_sessions)*100:.1f}%)")
        
        # Peak hours analysis by product
        product_hour_analysis = self.df.groupby(['coffee_name', 'hour_of_day']).size().unstack(fill_value=0)
        
        print(f"\n☕ Product Peak Hours:")
        for product in self.df['coffee_name'].value_counts().head().index:
            peak_hour = product_hour_analysis.loc[product].idxmax()
            peak_sales = product_hour_analysis.loc[product].max()
            print(f"   {product}: Peak at {peak_hour}:00 ({peak_sales} sales)")
        
        # Price sensitivity analysis
        price_volume_corr = self.df.groupby('coffee_name').agg({
            'money': ['mean', 'count']
        }).round(2)
        price_volume_corr.columns = ['avg_price', 'volume']
        correlation = price_volume_corr['avg_price'].corr(price_volume_corr['volume'])
        
        print(f"\n💰 Price-Volume Relationship:")
        print(f"   Price-Volume Correlation: {correlation:.3f}")
        if correlation < -0.3:
            print("   → Strong negative correlation: Higher prices = Lower volume")
        elif correlation > 0.3:
            print("   → Strong positive correlation: Premium products drive volume")
        else:
            print("   → Weak correlation: Price is not the primary driver")
    
    def analyze_payment_behaviors(self):
        """Analyze payment method preferences and behaviors"""
        print("\n💳 PAYMENT BEHAVIOR ANALYSIS:")
        print("-" * 40)
        
        # Payment method by amount
        payment_analysis = self.df.groupby('cash_type').agg({
            'money': ['count', 'mean', 'sum'],
            'hour_of_day': 'mean'
        }).round(2)
        payment_analysis.columns = ['transactions', 'avg_amount', 'total_spent', 'avg_hour']
        
        for method, stats in payment_analysis.iterrows():
            print(f"💳 {method.upper()}:")
            print(f"   Transactions: {stats['transactions']:,} ({stats['transactions']/len(self.df)*100:.1f}%)")
            print(f"   Average Amount: ${stats['avg_amount']:.2f}")
            print(f"   Total Volume: ${stats['total_spent']:,.2f}")
            print(f"   Peak Usage Hour: {stats['avg_hour']:.1f}")
        
        # Payment method by time of day
        payment_time = self.df.groupby(['time_of_day', 'cash_type']).size().unstack(fill_value=0)
        payment_time_pct = payment_time.div(payment_time.sum(axis=1), axis=0) * 100
        
        print(f"\n⏰ Payment Preferences by Time:")
        for time_period in payment_time_pct.index:
            print(f"   {time_period}:")
            for method in payment_time_pct.columns:
                pct = payment_time_pct.loc[time_period, method]
                print(f"     {method}: {pct:.1f}%")
    
    def create_behavioral_visualizations(self, customer_sessions, segment_summary):
        """Create comprehensive behavioral analysis visualizations"""
        
        # Create plotly dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Customer Segments by Spending', 'Purchase Patterns by Hour',
                          'Payment Method Preferences', 'Product Category Performance'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "box"}]]
        )
        
        # 1. Customer segments scatter plot
        colors = ['red', 'blue', 'green', 'orange']
        for seg in customer_sessions['segment'].unique():
            seg_data = customer_sessions[customer_sessions['segment'] == seg]
            segment_name = f"Segment {seg + 1}"
            
            fig.add_trace(
                go.Scatter(
                    x=seg_data['total_spent'],
                    y=seg_data['items_purchased'],
                    mode='markers',
                    name=segment_name,
                    marker=dict(color=colors[seg % len(colors)], size=8, opacity=0.6)
                ),
                row=1, col=1
            )
        
        # 2. Hourly purchase patterns
        hourly_patterns = self.df.groupby('hour_of_day').agg({
            'money': ['count', 'sum']
        })
        hourly_patterns.columns = ['transactions', 'revenue']
        
        fig.add_trace(
            go.Bar(
                x=hourly_patterns.index,
                y=hourly_patterns['transactions'],
                name='Transactions',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Payment method pie chart
        payment_dist = self.df['cash_type'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=payment_dist.index,
                values=payment_dist.values,
                name="Payment Methods"
            ),
            row=2, col=1
        )
        
        # 4. Product price distribution
        top_products = self.df['coffee_name'].value_counts().head(6).index
        
        for product in top_products:
            product_data = self.df[self.df['coffee_name'] == product]['money']
            fig.add_trace(
                go.Box(
                    y=product_data,
                    name=product,
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Customer Behavioral Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Total Spent ($)", row=1, col=1)
        fig.update_yaxes(title_text="Items Purchased", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Number of Transactions", row=1, col=2)
        fig.update_yaxes(title_text="Price ($)", row=2, col=2)
        
        # Save visualization
        fig.write_html('outputs/behavioral/behavioral_analysis_dashboard.html')
        
        print("💾 Behavioral analysis visualizations saved to outputs/behavioral/")
    
    def advanced_statistical_analysis(self):
        """Perform advanced statistical analysis and hypothesis testing"""
        print("\n" + "="*70)
        print("📊 ADVANCED STATISTICAL ANALYSIS & INSIGHTS")
        print("="*70)
        
        # Statistical tests and correlations
        print("\n🔬 STATISTICAL HYPOTHESIS TESTS:")
        print("-" * 40)
        
        # 1. Weekend vs Weekday spending test
        weekend_spending = self.df[self.df['is_weekend']]['money']
        weekday_spending = self.df[~self.df['is_weekend']]['money']
        
        t_stat, p_value = stats.ttest_ind(weekend_spending, weekday_spending)
        print(f"📈 Weekend vs Weekday Spending Test:")
        print(f"   Weekend Average: ${weekend_spending.mean():.2f}")
        print(f"   Weekday Average: ${weekday_spending.mean():.2f}")
        print(f"   T-statistic: {t_stat:.3f}")
        print(f"   P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("   → Significant difference detected!")
        else:
            print("   → No significant difference")
        
        # 2. Payment method and spending relationship
        payment_methods = self.df['cash_type'].unique()
        if len(payment_methods) > 1:
            payment_spending = [self.df[self.df['cash_type'] == method]['money'].values 
                               for method in payment_methods]
            f_stat, p_value = stats.f_oneway(*payment_spending)
            
            print(f"\n💳 Payment Method vs Spending ANOVA:")
            print(f"   F-statistic: {f_stat:.3f}")
            print(f"   P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("   → Payment method significantly affects spending!")
            else:
                print("   → No significant relationship")
        else:
            print(f"\n💳 Payment Method Analysis:")
            print(f"   Only one payment method detected: {payment_methods[0]}")
            print(f"   Cannot perform comparative analysis")
        
        # 3. Correlation analysis
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print("-" * 30)
        
        # Create correlation matrix for numerical variables
        numeric_cols = ['money', 'hour_of_day', 'day_of_week', 'day_of_month']
        correlation_matrix = self.df[numeric_cols].corr()
        
        print("Key Correlations:")
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_value = correlation_matrix.loc[col1, col2]
                print(f"   {col1} ↔ {col2}: {corr_value:.3f}")
        
        # Save correlation matrix
        correlation_matrix.to_csv('outputs/advanced/correlation_matrix.csv')
        
        # 4. Product preference by time analysis
        print(f"\n⏰ TEMPORAL PRODUCT PREFERENCES:")
        print("-" * 35)
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(self.df['time_of_day'], self.df['coffee_name'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"   Chi-square statistic: {chi2:.3f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print("   → Product preferences vary significantly by time of day!")
        else:
            print("   → No significant temporal preference pattern")
        
        # 5. Revenue anomaly detection
        self.detect_revenue_anomalies()
        
        return correlation_matrix
    
    def detect_revenue_anomalies(self):
        """Detect anomalous days in revenue using statistical methods"""
        print(f"\n🚨 REVENUE ANOMALY DETECTION:")
        print("-" * 35)
        
        # Daily revenue calculation
        daily_revenue = self.df.groupby('date')['money'].sum()
        
        # Statistical anomaly detection using Z-score
        mean_revenue = daily_revenue.mean()
        std_revenue = daily_revenue.std()
        z_scores = np.abs((daily_revenue - mean_revenue) / std_revenue)
        
        # Identify anomalies (Z-score > 2)
        anomalies = daily_revenue[z_scores > 2]
        
        print(f"   Average Daily Revenue: ${mean_revenue:.2f}")
        print(f"   Revenue Standard Deviation: ${std_revenue:.2f}")
        print(f"   Anomalous Days Detected: {len(anomalies)}")
        
        if len(anomalies) > 0:
            print(f"\n🔍 Anomalous Days:")
            for date, revenue in anomalies.items():
                z_score = z_scores[date]
                day_of_week = date.strftime('%A')
                print(f"   {date.strftime('%Y-%m-%d')} ({day_of_week}): ${revenue:.2f} (Z-score: {z_score:.2f})")
            
            # Save anomalies
            anomaly_df = pd.DataFrame({
                'date': anomalies.index,
                'revenue': anomalies.values,
                'z_score': [z_scores[date] for date in anomalies.index],
                'day_of_week': [date.strftime('%A') for date in anomalies.index]
            })
            anomaly_df.to_csv('outputs/advanced/revenue_anomalies.csv', index=False)
        
        # IQR method for comparison
        Q1 = daily_revenue.quantile(0.25)
        Q3 = daily_revenue.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = daily_revenue[(daily_revenue < lower_bound) | (daily_revenue > upper_bound)]
        print(f"   IQR Method Anomalies: {len(iqr_anomalies)} days")
    
    def generate_business_insights_report(self):
        """Generate comprehensive business insights and recommendations"""
        print("\n" + "="*70)
        print("📋 COMPREHENSIVE BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*70)
        
        # Key performance indicators
        total_revenue = self.df['money'].sum()
        avg_daily_revenue = self.df.groupby('date')['money'].sum().mean()
        total_transactions = len(self.df)
        avg_transaction_value = self.df['money'].mean()
        operating_days = self.df['date'].nunique()
        
        print(f"\n📈 KEY BUSINESS METRICS:")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Average Daily Revenue: ${avg_daily_revenue:.2f}")
        print(f"   Total Transactions: {total_transactions:,}")
        print(f"   Average Transaction Value: ${avg_transaction_value:.2f}")
        print(f"   Operating Days: {operating_days}")
        print(f"   Revenue per Operating Day: ${total_revenue/operating_days:.2f}")
        
        # Growth opportunities
        print(f"\n🚀 GROWTH OPPORTUNITIES:")
        
        # Peak hours analysis
        hourly_revenue = self.df.groupby('hour_of_day')['money'].sum()
        peak_hours = hourly_revenue.nlargest(3)
        off_peak_hours = hourly_revenue.nsmallest(3)
        
        print(f"   Peak Revenue Hours: {', '.join([f'{h}:00' for h in peak_hours.index])}")
        print(f"   Underperforming Hours: {', '.join([f'{h}:00' for h in off_peak_hours.index])}")
        
        # Product opportunities
        product_performance = self.df.groupby('coffee_name').agg({
            'money': ['count', 'sum', 'mean']
        })
        product_performance.columns = ['frequency', 'revenue', 'avg_price']
        
        # High margin, low frequency products (opportunities)
        high_margin_low_freq = product_performance[
            (product_performance['avg_price'] > product_performance['avg_price'].median()) &
            (product_performance['frequency'] < product_performance['frequency'].median())
        ]
        
        if len(high_margin_low_freq) > 0:
            print(f"   High-Margin Growth Products:")
            for product in high_margin_low_freq.index[:3]:
                avg_price = high_margin_low_freq.loc[product, 'avg_price']
                frequency = high_margin_low_freq.loc[product, 'frequency']
                print(f"     • {product}: ${avg_price:.2f} avg (only {frequency} sales)")
        
        # Strategic recommendations
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        
        # 1. Staffing optimization
        peak_hour = hourly_revenue.idxmax()
        print(f"   1. STAFFING: Increase staff during {peak_hour}:00 hour")
        
        # 2. Product strategy
        top_product = self.df.groupby('coffee_name')['money'].sum().idxmax()
        print(f"   2. PRODUCT FOCUS: Promote '{top_product}' as flagship product")
        
        # 3. Pricing strategy
        if self.df['money'].std() > 5:  # High price variation
            print(f"   3. PRICING: Consider dynamic pricing during peak hours")
        
        # 4. Customer retention
        weekend_revenue = self.df[self.df['is_weekend']]['money'].sum()
        weekday_revenue = self.df[~self.df['is_weekend']]['money'].sum()
        
        if weekend_revenue < weekday_revenue * 0.3:  # Low weekend performance
            print(f"   4. MARKETING: Develop weekend promotions and events")
        
        # 5. Payment optimization
        card_transactions = len(self.df[self.df['cash_type'] == 'card'])
        cash_transactions = len(self.df[self.df['cash_type'] == 'cash'])
        
        if card_transactions > cash_transactions:
            print(f"   5. PAYMENT: Consider contactless payment incentives")
        
        # Save comprehensive report
        report_data = {
            'metric': ['Total Revenue', 'Average Daily Revenue', 'Total Transactions', 
                      'Average Transaction Value', 'Peak Revenue Hour', 'Top Product'],
            'value': [f"${total_revenue:,.2f}", f"${avg_daily_revenue:.2f}", 
                     f"{total_transactions:,}", f"${avg_transaction_value:.2f}",
                     f"{peak_hour}:00", top_product]
        }
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv('outputs/advanced/business_insights_report.csv', index=False)
        
        print(f"\n💾 Reports saved to outputs/advanced/")
        print(f"   • business_insights_report.csv")
        print(f"   • correlation_matrix.csv")
        print(f"   • revenue_anomalies.csv")
    
    def run_complete_analysis(self):
        """Execute the complete advanced analytics suite"""
        print("🚀 Starting Complete Advanced Analytics Suite...")
        print("="*70)
        
        # Run all analysis modules
        revenue_results, count_results = self.demand_forecasting_suite()
        customer_sessions, segments = self.customer_behavioral_analysis()
        correlation_matrix = self.advanced_statistical_analysis()
        self.generate_business_insights_report()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ADVANCED ANALYTICS COMPLETE!")
        print("="*70)
        
        print("\n📁 Generated Files:")
        print("   PREDICTIONS:")
        print("     • outputs/predictions/30_day_forecasts.csv")
        print("     • outputs/predictions/prediction_analysis.html")
        
        print("   BEHAVIORAL ANALYSIS:")
        print("     • outputs/behavioral/customer_segments.csv") 
        print("     • outputs/behavioral/behavioral_analysis_dashboard.html")
        
        print("   ADVANCED ANALYTICS:")
        print("     • outputs/advanced/business_insights_report.csv")
        print("     • outputs/advanced/correlation_matrix.csv")
        print("     • outputs/advanced/revenue_anomalies.csv")
        
        print(f"\n🎯 KEY INSIGHTS SUMMARY:")
        print(f"   • {len(customer_sessions):,} customer sessions analyzed")
        print(f"   • {len(segments)} customer segments identified")
        print(f"   • 30-day demand forecast generated")
        print(f"   • Revenue anomalies detected and analyzed")
        print(f"   • Interactive dashboards created")
        
        print(f"\n🚀 Ready for data-driven business optimization!")

def main():
    """Main execution function"""
    print("☕ ADVANCED COFFEE SHOP ANALYTICS ENGINE")
    print("="*50)
    print("Features:")
    print("• Demand Forecasting with ML Models")
    print("• Customer Behavioral Segmentation") 
    print("• Statistical Hypothesis Testing")
    print("• Anomaly Detection")
    print("• Interactive Visualizations")
    print("• Business Intelligence Reports")
    print("="*50)
    
    # Initialize and run analytics
    analytics_engine = CoffeeShopAdvancedAnalytics()
    analytics_engine.run_complete_analysis()
    
    print("\n🎉 Advanced Analytics Suite Execution Complete!")

if __name__ == "__main__":
    main()