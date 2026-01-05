"""Product Preference Analysis Module (ARD Section 4.3).

Business Questions:
- Which coffee products are most popular?
- Do product preferences vary by time or day?
- Are certain products time-specific?

Analysis Scope:
- Product sales volume and revenue
- Product performance by Time_of_Day
- Product performance by hour and weekday
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class ProductPreferenceAnalyzer:
    """Analyzes product preferences and patterns."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ProductPreferenceAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.results: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def analyze_product_popularity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze overall product popularity.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with product popularity metrics
        """
        self.logger.info("Analyzing product popularity...")
        
        if 'coffee_name' not in df.columns:
            raise ValueError("'coffee_name' column not found")
        
        product_metrics = df.groupby('coffee_name').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        product_metrics.columns = ['total_revenue', 'avg_price', 'sales_count']
        
        # Calculate market share
        product_metrics['revenue_share'] = (
            product_metrics['total_revenue'] / product_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        product_metrics['volume_share'] = (
            product_metrics['sales_count'] / product_metrics['sales_count'].sum() * 100
        ).round(2)
        
        # Rank products
        product_metrics['popularity_rank'] = product_metrics['sales_count'].rank(ascending=False, method='dense').astype(int)
        product_metrics['revenue_rank'] = product_metrics['total_revenue'].rank(ascending=False, method='dense').astype(int)
        
        product_metrics = product_metrics.reset_index()
        product_metrics = product_metrics.sort_values('sales_count', ascending=False)
        
        self.logger.info(f"Analyzed {len(product_metrics)} products")
        self.logger.info(f"Most popular: {product_metrics.iloc[0]['coffee_name']} ({int(product_metrics.iloc[0]['sales_count'])} sales)")
        
        return product_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_product_by_time_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze product preferences by time of day.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with product-time relationships
        """
        self.logger.info("Analyzing products by time of day...")
        
        if 'coffee_name' not in df.columns or 'Time_of_Day' not in df.columns:
            self.logger.warning("Required columns not found, skipping...")
            return pd.DataFrame()
        
        product_time = df.groupby(['coffee_name', 'Time_of_Day']).agg({
            'money': ['sum', 'count']
        }).round(2)
        
        product_time.columns = ['total_revenue', 'sales_count']
        
        # Calculate percentage of product's total sales in each time period
        product_totals = df.groupby('coffee_name').size()
        product_time['pct_of_product_sales'] = product_time.groupby(level=0)['sales_count'].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )
        
        product_time = product_time.reset_index()
        
        # Find dominant time for each product
        dominant_time = product_time.loc[product_time.groupby('coffee_name')['sales_count'].idxmax()]
        dominant_time = dominant_time[['coffee_name', 'Time_of_Day']].rename(
            columns={'Time_of_Day': 'dominant_time_period'}
        )
        
        product_time = product_time.merge(dominant_time, on='coffee_name', how='left')
        
        self.logger.info(f"Analyzed {len(product_time)} product-time combinations")
        
        return product_time
    
    @log_execution_time(setup_logger(__name__))
    def analyze_product_by_hour(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze top products by hour of day.
        
        Args:
            df: Sales DataFrame
            top_n: Number of top products to analyze
            
        Returns:
            DataFrame with product-hour patterns
        """
        self.logger.info(f"Analyzing top {top_n} products by hour...")
        
        if 'coffee_name' not in df.columns or 'hour_of_day' not in df.columns:
            raise ValueError("Required columns not found")
        
        # Get top N products by sales count
        top_products = df['coffee_name'].value_counts().head(top_n).index.tolist()
        
        df_top = df[df['coffee_name'].isin(top_products)]
        
        product_hour = df_top.groupby(['coffee_name', 'hour_of_day']).size().reset_index(name='sales_count')
        
        # Create pivot for easier analysis
        product_hour_pivot = product_hour.pivot(
            index='coffee_name',
            columns='hour_of_day',
            values='sales_count'
        ).fillna(0).astype(int)
        
        # Find peak hour for each product
        peak_hours = product_hour_pivot.idxmax(axis=1).reset_index()
        peak_hours.columns = ['coffee_name', 'peak_hour']
        
        self.logger.info(f"Analyzed hourly patterns for {len(top_products)} products")
        
        return product_hour_pivot
    
    @log_execution_time(setup_logger(__name__))
    def analyze_product_by_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze product preferences by weekday.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with product-weekday patterns
        """
        self.logger.info("Analyzing products by weekday...")
        
        if 'coffee_name' not in df.columns or 'Weekday' not in df.columns:
            raise ValueError("Required columns not found")
        
        product_weekday = df.groupby(['coffee_name', 'Weekday']).agg({
            'money': ['sum', 'count']
        }).round(2)
        
        product_weekday.columns = ['total_revenue', 'sales_count']
        
        # Calculate percentage of product's weekly sales on each day
        product_weekday['pct_of_product_sales'] = product_weekday.groupby(level=0)['sales_count'].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )
        
        product_weekday = product_weekday.reset_index()
        
        # Find dominant weekday for each product
        dominant_day = product_weekday.loc[product_weekday.groupby('coffee_name')['sales_count'].idxmax()]
        dominant_day = dominant_day[['coffee_name', 'Weekday']].rename(
            columns={'Weekday': 'dominant_weekday'}
        )
        
        product_weekday = product_weekday.merge(dominant_day, on='coffee_name', how='left')
        
        self.logger.info(f"Analyzed {len(product_weekday)} product-weekday combinations")
        
        return product_weekday
    
    @log_execution_time(setup_logger(__name__))
    def identify_time_specific_products(self, df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
        """
        Identify products that are time-specific (concentrated in certain periods).
        
        Args:
            df: Sales DataFrame
            threshold: Percentage threshold to consider a product time-specific
            
        Returns:
            DataFrame with time-specific products
        """
        self.logger.info(f"Identifying time-specific products (>{threshold}% in one period)...")
        
        if 'coffee_name' not in df.columns or 'Time_of_Day' not in df.columns:
            self.logger.warning("Required columns not found, skipping...")
            return pd.DataFrame()
        
        product_time = df.groupby(['coffee_name', 'Time_of_Day']).size().reset_index(name='sales_count')
        
        # Calculate percentage by product
        product_totals = product_time.groupby('coffee_name')['sales_count'].transform('sum')
        product_time['pct_in_period'] = (product_time['sales_count'] / product_totals * 100).round(2)
        
        # Find maximum percentage for each product
        max_pct = product_time.groupby('coffee_name')['pct_in_period'].max().reset_index()
        max_pct.columns = ['coffee_name', 'max_concentration']
        
        # Identify time-specific products
        time_specific = max_pct[max_pct['max_concentration'] >= threshold]
        
        # Get the dominant time period
        dominant = product_time.loc[product_time.groupby('coffee_name')['pct_in_period'].idxmax()]
        time_specific = time_specific.merge(
            dominant[['coffee_name', 'Time_of_Day', 'pct_in_period']],
            on='coffee_name'
        )
        time_specific = time_specific.rename(columns={'Time_of_Day': 'dominant_period'})
        
        time_specific = time_specific.sort_values('pct_in_period', ascending=False)
        
        self.logger.info(f"Found {len(time_specific)} time-specific products")
        
        return time_specific
    
    @log_execution_time(setup_logger(__name__))
    def run_analysis(self, df: pd.DataFrame, save_outputs: bool = True) -> Dict:
        """
        Execute complete product preference analysis.
        
        Args:
            df: Sales DataFrame
            save_outputs: Whether to save outputs to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("PRODUCT PREFERENCE ANALYSIS (ARD Section 4.3)")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Overall popularity
        results['product_popularity'] = self.analyze_product_popularity(df)
        
        # Product by time of day
        results['product_by_time_of_day'] = self.analyze_product_by_time_of_day(df)
        
        # Product by hour
        results['product_by_hour'] = self.analyze_product_by_hour(df, top_n=10)
        
        # Product by weekday
        results['product_by_weekday'] = self.analyze_product_by_weekday(df)
        
        # Time-specific products
        results['time_specific_products'] = self.identify_time_specific_products(df, threshold=50.0)
        
        self.results = results
        
        # Save outputs if requested
        if save_outputs:
            self.save_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Product Preference Analysis Complete")
        self.logger.info("=" * 80)
        
        return results
    
    def save_results(self, results: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Save analysis results to CSV files.
        
        Args:
            results: Results dictionary. If None, uses self.results.
            
        Returns:
            Dictionary mapping result names to saved file paths
        """
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results to save")
        
        self.logger.info("Saving product preference analysis results...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save DataFrames
        dataframe_outputs = {
            'product_popularity': 'coffee_counts.csv',
            'product_by_time_of_day': 'product_time_of_day_analysis.csv',
            'product_by_hour': 'product_hourly_patterns.csv',
            'product_by_weekday': 'product_weekday_patterns.csv',
            'time_specific_products': 'time_specific_products.csv'
        }
        
        for key, filename in dataframe_outputs.items():
            if key in results and isinstance(results[key], pd.DataFrame) and not results[key].empty:
                filepath = output_dir / filename
                if key == 'product_by_hour':
                    results[key].to_csv(filepath, index=True)
                else:
                    results[key].to_csv(filepath, index=False)
                saved_files[key] = filepath
                self.logger.info(f"Saved {key} to {filepath}")
        
        return saved_files
    
    def get_insights(self, results: Optional[Dict] = None) -> List[str]:
        """
        Generate business insights from analysis results.
        
        Args:
            results: Results dictionary. If None, uses self.results.
            
        Returns:
            List of insight strings
        """
        if results is None:
            results = self.results
        
        if not results:
            return []
        
        insights = []
        
        # Popularity insights
        if 'product_popularity' in results:
            products = results['product_popularity']
            if not products.empty:
                top_product = products.iloc[0]
                insights.append(
                    f"Most popular product: {top_product['coffee_name']} "
                    f"({int(top_product['sales_count'])} sales, {top_product['volume_share']:.1f}% market share)"
                )
                
                # Concentration analysis
                top5_share = products.head(5)['volume_share'].sum()
                insights.append(
                    f"Top 5 products account for {top5_share:.1f}% of total sales volume"
                )
        
        # Time-specific insights
        if 'time_specific_products' in results and not results['time_specific_products'].empty:
            time_specific = results['time_specific_products']
            top_specific = time_specific.iloc[0]
            insights.append(
                f"Most time-specific product: {top_specific['coffee_name']} "
                f"({top_specific['pct_in_period']:.1f}% sold during {top_specific['dominant_period']})"
            )
        
        # Product-time alignment
        if 'product_by_time_of_day' in results and not results['product_by_time_of_day'].empty:
            prod_time = results['product_by_time_of_day']
            morning_products = prod_time[prod_time['dominant_time_period'] == 'Morning']['coffee_name'].nunique()
            total_products = prod_time['coffee_name'].nunique()
            insights.append(
                f"{morning_products} out of {total_products} products are primarily sold in the Morning"
            )
        
        return insights
