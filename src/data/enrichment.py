"""Data enrichment module for Coffee Sales Analysis.

This module handles:
- Feature engineering based on ARD requirements
- Time-based features (peak hours, time segments)
- Revenue features (price analysis, transaction patterns)
- Behavioral features (payment patterns, product preferences)
- Advanced derived metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_dataframe_info, log_execution_time


class DataEnrichment:
    """Enriches preprocessed coffee sales data with derived features."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize DataEnrichment.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.enriched_data: Optional[pd.DataFrame] = None
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for demand analysis (ARD Section 4.2).
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with time features
        """
        self.logger.info("Adding time-based features...")
        df = df.copy()
        
        # Hour-based features
        if 'hour_of_day' in df.columns:
            # Define time segments
            df['hour_segment'] = pd.cut(
                df['hour_of_day'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # Peak hour indicator (based on typical coffee shop patterns)
            df['is_peak_hour'] = df['hour_of_day'].isin([7, 8, 9, 10, 12, 13, 14, 15]).astype(int)
            
            # Rush hour (morning and lunch)
            df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 12, 13]).astype(int)
        
        # Weekday features
        if 'Weekday' in df.columns:
            # Map weekday names to categories
            weekday_mapping = {
                'Monday': 'Weekday',
                'Tuesday': 'Weekday',
                'Wednesday': 'Weekday',
                'Thursday': 'Weekday',
                'Friday': 'Weekday',
                'Saturday': 'Weekend',
                'Sunday': 'Weekend'
            }
            df['day_type'] = df['Weekday'].map(weekday_mapping)
        
        # Date-based features (if not already added)
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            if 'quarter' not in df.columns:
                df['quarter'] = df['Date'].dt.quarter
            
            if 'is_month_start' not in df.columns:
                df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
            
            if 'is_month_end' not in df.columns:
                df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        self.logger.info(f"Time features added - Total columns: {len(df.columns)}")
        
        return df
    
    def add_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add revenue and pricing features (ARD Section 4.1).
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with revenue features
        """
        self.logger.info("Adding revenue features...")
        df = df.copy()
        
        if 'money' not in df.columns:
            self.logger.warning("'money' column not found. Skipping revenue features.")
            return df
        
        # Basic revenue metrics
        df['transaction_value'] = df['money']
        
        # Revenue categories
        revenue_percentiles = df['money'].quantile([0.25, 0.5, 0.75])
        df['revenue_category'] = pd.cut(
            df['money'],
            bins=[0, revenue_percentiles[0.25], revenue_percentiles[0.5], 
                  revenue_percentiles[0.75], df['money'].max()],
            labels=['Low', 'Medium', 'High', 'Premium'],
            include_lowest=True
        )
        
        # Product price analysis
        if 'coffee_name' in df.columns:
            # Average price per product
            product_avg_price = df.groupby('coffee_name')['money'].transform('mean')
            df['product_avg_price'] = product_avg_price
            
            # Price deviation from product average
            df['price_deviation'] = df['money'] - df['product_avg_price']
            df['price_deviation_pct'] = (df['price_deviation'] / df['product_avg_price'] * 100).round(2)
            
            # Product price rank
            df['product_price_rank'] = df.groupby('coffee_name')['money'].rank(method='dense')
        
        # Daily aggregations
        if 'Date' in df.columns:
            # Daily revenue
            daily_revenue = df.groupby('Date')['money'].transform('sum')
            df['daily_total_revenue'] = daily_revenue
            
            # Transaction contribution to daily revenue
            df['revenue_share_of_day'] = (df['money'] / df['daily_total_revenue'] * 100).round(2)
        
        # Hourly aggregations
        if 'hour_of_day' in df.columns and 'Date' in df.columns:
            # Hourly revenue
            hourly_revenue = df.groupby(['Date', 'hour_of_day'])['money'].transform('sum')
            df['hourly_total_revenue'] = hourly_revenue
        
        self.logger.info(f"Revenue features added - Total columns: {len(df.columns)}")
        
        return df
    
    def add_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add product preference features (ARD Section 4.3).
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with product features
        """
        self.logger.info("Adding product features...")
        df = df.copy()
        
        if 'coffee_name' not in df.columns:
            self.logger.warning("'coffee_name' column not found. Skipping product features.")
            return df
        
        # Product popularity metrics
        product_counts = df['coffee_name'].value_counts()
        df['product_total_sales'] = df['coffee_name'].map(product_counts)
        
        # Product popularity rank
        product_rank = product_counts.rank(method='dense', ascending=False)
        df['product_popularity_rank'] = df['coffee_name'].map(product_rank).astype(int)
        
        # Product category based on popularity
        total_products = df['coffee_name'].nunique()
        df['product_tier'] = pd.cut(
            df['product_popularity_rank'],
            bins=[0, total_products * 0.2, total_products * 0.5, total_products],
            labels=['Top', 'Mid', 'Long-tail'],
            include_lowest=True
        )
        
        # Time-specific product features
        if 'Time_of_Day' in df.columns:
            # Product sales by time of day
            product_time_counts = df.groupby(['coffee_name', 'Time_of_Day']).size()
            df['product_time_sales'] = df.apply(
                lambda row: product_time_counts.get((row['coffee_name'], row['Time_of_Day']), 0),
                axis=1
            )
        
        # Weekday-specific product features
        if 'Weekday' in df.columns:
            # Most common weekday for each product
            product_weekday = df.groupby('coffee_name')['Weekday'].agg(
                lambda x: x.value_counts().index[0]
            )
            df['product_peak_weekday'] = df['coffee_name'].map(product_weekday)
        
        self.logger.info(f"Product features added - Total columns: {len(df.columns)}")
        
        return df
    
    def add_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add payment behavior features (ARD Section 4.4).
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with payment features
        """
        self.logger.info("Adding payment features...")
        df = df.copy()
        
        if 'cash_type' not in df.columns:
            self.logger.warning("'cash_type' column not found. Skipping payment features.")
            return df
        
        # Payment type encoding
        df['is_cash_payment'] = (df['cash_type'].str.lower() == 'cash').astype(int)
        
        # Average spend by payment type
        if 'money' in df.columns:
            payment_avg_spend = df.groupby('cash_type')['money'].transform('mean')
            df['payment_type_avg_spend'] = payment_avg_spend
            
            # Deviation from payment type average
            df['spend_vs_payment_avg'] = df['money'] - df['payment_type_avg_spend']
        
        # Payment method frequency
        payment_counts = df['cash_type'].value_counts()
        df['payment_method_frequency'] = df['cash_type'].map(payment_counts)
        
        # Time-based payment patterns
        if 'hour_of_day' in df.columns:
            # Payment preference by hour
            hourly_payment = df.groupby('hour_of_day')['is_cash_payment'].transform('mean')
            df['hourly_cash_rate'] = (hourly_payment * 100).round(2)
        
        self.logger.info(f"Payment features added - Total columns: {len(df.columns)}")
        
        return df
    
    def add_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add customer behavior and pattern features.
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with behavioral features
        """
        self.logger.info("Adding customer behavior features...")
        df = df.copy()
        
        # Transaction sequence features
        if 'Date' in df.columns and 'hour_of_day' in df.columns:
            # Sort by datetime
            df = df.sort_values(['Date', 'hour_of_day'])
            
            # Transaction number within day
            df['transaction_num_in_day'] = df.groupby('Date').cumcount() + 1
            
            # Total transactions per day
            df['total_transactions_in_day'] = df.groupby('Date')['Date'].transform('count')
        
        # Repeat pattern features
        if 'coffee_name' in df.columns and 'Date' in df.columns:
            # Product variety per day
            daily_product_variety = df.groupby('Date')['coffee_name'].transform('nunique')
            df['daily_product_variety'] = daily_product_variety
        
        # Spending pattern features
        if 'money' in df.columns and 'Date' in df.columns:
            # Rolling average spend (7-day window)
            df['rolling_7day_avg_spend'] = df.groupby('Date')['money'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
        
        self.logger.info(f"Behavioral features added - Total columns: {len(df.columns)}")
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical and derived metrics.
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            DataFrame with statistical features
        """
        self.logger.info("Adding statistical features...")
        df = df.copy()
        
        if 'money' in df.columns:
            # Z-score for transaction value
            df['transaction_value_zscore'] = (
                df['money'] - df['money'].mean()
            ) / df['money'].std()
            
            # Percentile rank
            df['transaction_value_percentile'] = df['money'].rank(pct=True) * 100
            
            # Outlier detection (using IQR method)
            Q1 = df['money'].quantile(0.25)
            Q3 = df['money'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df['is_outlier'] = ((df['money'] < lower_bound) | (df['money'] > upper_bound)).astype(int)
        
        self.logger.info(f"Statistical features added - Total columns: {len(df.columns)}")
        
        return df
    
    @log_execution_time(setup_logger(__name__))
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full enrichment pipeline.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Enriched DataFrame
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting data enrichment pipeline")
        self.logger.info("=" * 80)
        
        initial_columns = len(df.columns)
        
        # Add all feature groups
        df = self.add_time_features(df)
        df = self.add_revenue_features(df)
        df = self.add_product_features(df)
        df = self.add_payment_features(df)
        df = self.add_customer_behavior_features(df)
        df = self.add_statistical_features(df)
        
        self.enriched_data = df
        
        new_features = len(df.columns) - initial_columns
        self.logger.info("=" * 80)
        self.logger.info(f"Enrichment complete - Added {new_features} new features")
        self.logger.info(f"Total columns: {len(df.columns)}")
        self.logger.info("=" * 80)
        
        return df
    
    def save_enriched_data(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Save enriched data to CSV.
        
        Args:
            df: DataFrame to save. If None, uses enriched_data.
            output_path: Output file path. If None, uses config path.
            
        Returns:
            Path to saved file
        """
        if df is None:
            df = self.enriched_data
        
        if df is None:
            raise ValueError("No enriched data to save")
        
        if output_path is None:
            output_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'enriched_coffee_sales.csv'
        else:
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving enriched data to {output_path}")
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Successfully saved {len(df)} rows with {len(df.columns)} columns to {output_path}")
        
        return output_path
    
    def get_feature_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get summary of all features in the enriched dataset.
        
        Args:
            df: DataFrame to summarize. If None, uses enriched_data.
            
        Returns:
            Dictionary with feature categories and descriptions
        """
        if df is None:
            df = self.enriched_data
        
        if df is None:
            raise ValueError("No data to summarize")
        
        feature_summary = {
            'total_features': len(df.columns),
            'feature_categories': {
                'original_features': [],
                'time_features': [],
                'revenue_features': [],
                'product_features': [],
                'payment_features': [],
                'behavioral_features': [],
                'statistical_features': []
            }
        }
        
        # Categorize features based on naming patterns
        for col in df.columns:
            col_lower = col.lower()
            
            if any(x in col_lower for x in ['hour', 'day', 'week', 'month', 'year', 'time', 'date', 'quarter', 'peak', 'rush']):
                feature_summary['feature_categories']['time_features'].append(col)
            elif any(x in col_lower for x in ['revenue', 'price', 'money', 'transaction_value']):
                feature_summary['feature_categories']['revenue_features'].append(col)
            elif any(x in col_lower for x in ['product', 'coffee']):
                feature_summary['feature_categories']['product_features'].append(col)
            elif any(x in col_lower for x in ['payment', 'cash']):
                feature_summary['feature_categories']['payment_features'].append(col)
            elif any(x in col_lower for x in ['behavior', 'pattern', 'sequence', 'variety']):
                feature_summary['feature_categories']['behavioral_features'].append(col)
            elif any(x in col_lower for x in ['zscore', 'percentile', 'outlier', 'rolling']):
                feature_summary['feature_categories']['statistical_features'].append(col)
            else:
                feature_summary['feature_categories']['original_features'].append(col)
        
        return feature_summary
