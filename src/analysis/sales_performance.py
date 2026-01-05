"""Sales Performance Analysis Module (ARD Section 4.1).

Business Questions:
- How much total revenue is generated?
- Which coffee products contribute the most and least revenue?
- How does revenue change over time?

Analysis Scope:
- Total revenue and transaction count
- Revenue by coffee type
- Revenue by day, weekday, and month
- Average transaction value
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class SalesPerformanceAnalyzer:
    """Analyzes sales performance and revenue metrics."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize SalesPerformanceAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.results: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def analyze_total_revenue(self, df: pd.DataFrame) -> Dict:
        """
        Calculate total revenue and transaction metrics.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with total revenue metrics
        """
        self.logger.info("Analyzing total revenue metrics...")
        
        if 'money' not in df.columns:
            raise ValueError("'money' column not found in DataFrame")
        
        metrics = {
            'total_revenue': float(df['money'].sum()),
            'total_transactions': int(len(df)),
            'average_transaction_value': float(df['money'].mean()),
            'median_transaction_value': float(df['money'].median()),
            'std_transaction_value': float(df['money'].std()),
            'min_transaction': float(df['money'].min()),
            'max_transaction': float(df['money'].max()),
            'revenue_range': float(df['money'].max() - df['money'].min())
        }
        
        self.logger.info(f"Total Revenue: ${metrics['total_revenue']:,.2f}")
        self.logger.info(f"Total Transactions: {metrics['total_transactions']:,}")
        self.logger.info(f"Average Transaction: ${metrics['average_transaction_value']:.2f}")
        
        return metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_revenue_by_product(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze revenue contribution by coffee product.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with product performance metrics
        """
        self.logger.info("Analyzing revenue by product...")
        
        if 'coffee_name' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        product_metrics = df.groupby('coffee_name').agg({
            'money': ['sum', 'mean', 'median', 'count', 'std']
        }).round(2)
        
        product_metrics.columns = [
            'total_revenue', 'avg_price', 'median_price', 
            'transaction_count', 'price_std'
        ]
        
        # Calculate revenue percentage
        product_metrics['revenue_percentage'] = (
            product_metrics['total_revenue'] / product_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        # Calculate cumulative revenue percentage
        product_metrics = product_metrics.sort_values('total_revenue', ascending=False)
        product_metrics['cumulative_revenue_pct'] = product_metrics['revenue_percentage'].cumsum().round(2)
        
        # Add product rank
        product_metrics['revenue_rank'] = range(1, len(product_metrics) + 1)
        
        # Identify top and bottom performers
        product_metrics['performance_tier'] = pd.cut(
            product_metrics['revenue_rank'],
            bins=[0, 5, 10, float('inf')],
            labels=['Top', 'Mid', 'Long-tail']
        )
        
        product_metrics = product_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(product_metrics)} products")
        self.logger.info(f"Top product: {product_metrics.iloc[0]['coffee_name']} "
                        f"(${product_metrics.iloc[0]['total_revenue']:,.2f})")
        
        return product_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_daily_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze revenue trends by date.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with daily revenue metrics
        """
        self.logger.info("Analyzing daily revenue trends...")
        
        if 'Date' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        daily_metrics = df.groupby('Date').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        daily_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Add day of week
        daily_metrics['day_of_week'] = daily_metrics.index.dayofweek
        daily_metrics['day_name'] = daily_metrics.index.day_name()
        
        # Calculate rolling averages
        daily_metrics['revenue_7day_ma'] = daily_metrics['total_revenue'].rolling(window=7, min_periods=1).mean().round(2)
        daily_metrics['revenue_30day_ma'] = daily_metrics['total_revenue'].rolling(window=30, min_periods=1).mean().round(2)
        
        # Calculate growth metrics
        daily_metrics['revenue_daily_change'] = daily_metrics['total_revenue'].diff().round(2)
        daily_metrics['revenue_pct_change'] = daily_metrics['total_revenue'].pct_change().round(4) * 100
        
        daily_metrics = daily_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(daily_metrics)} days")
        self.logger.info(f"Average daily revenue: ${daily_metrics['total_revenue'].mean():,.2f}")
        
        return daily_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_weekday_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze revenue patterns by weekday.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with weekday revenue metrics
        """
        self.logger.info("Analyzing revenue by weekday...")
        
        if 'Weekday' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        # Define weekday order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekday_metrics = df.groupby('Weekday').agg({
            'money': ['sum', 'mean', 'median', 'count']
        }).round(2)
        
        weekday_metrics.columns = ['total_revenue', 'avg_transaction', 'median_transaction', 'transaction_count']
        
        # Calculate percentage of total revenue
        weekday_metrics['revenue_percentage'] = (
            weekday_metrics['total_revenue'] / weekday_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        weekday_metrics = weekday_metrics.reset_index()
        
        # Sort by weekday order
        weekday_metrics['day_order'] = weekday_metrics['Weekday'].map({day: i for i, day in enumerate(weekday_order)})
        weekday_metrics = weekday_metrics.sort_values('day_order').drop('day_order', axis=1)
        
        self.logger.info(f"Highest revenue day: {weekday_metrics.iloc[weekday_metrics['total_revenue'].idxmax()]['Weekday']}")
        
        return weekday_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_monthly_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze revenue trends by month.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with monthly revenue metrics
        """
        self.logger.info("Analyzing monthly revenue trends...")
        
        if 'Month_name' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        monthly_metrics = df.groupby('Month_name').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        monthly_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentage
        monthly_metrics['revenue_percentage'] = (
            monthly_metrics['total_revenue'] / monthly_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        # Add month order if available
        if 'Monthsort' in df.columns:
            month_order = df.groupby('Month_name')['Monthsort'].first().to_dict()
            monthly_metrics['month_order'] = monthly_metrics.index.map(month_order)
            monthly_metrics = monthly_metrics.sort_values('month_order')
        
        monthly_metrics = monthly_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(monthly_metrics)} months")
        
        return monthly_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_price_by_product(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detailed price analysis by coffee product.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with price statistics by product
        """
        self.logger.info("Analyzing price distribution by product...")
        
        if 'coffee_name' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        price_stats = df.groupby('coffee_name')['money'].agg([
            ('min_price', 'min'),
            ('max_price', 'max'),
            ('avg_price', 'mean'),
            ('median_price', 'median'),
            ('std_price', 'std'),
            ('price_range', lambda x: x.max() - x.min())
        ]).round(2)
        
        price_stats = price_stats.reset_index()
        price_stats = price_stats.sort_values('avg_price', ascending=False)
        
        self.logger.info(f"Price analysis complete for {len(price_stats)} products")
        
        return price_stats
    
    @log_execution_time(setup_logger(__name__))
    def run_analysis(self, df: pd.DataFrame, save_outputs: bool = True) -> Dict:
        """
        Execute complete sales performance analysis.
        
        Args:
            df: Sales DataFrame
            save_outputs: Whether to save outputs to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("SALES PERFORMANCE ANALYSIS (ARD Section 4.1)")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Total revenue metrics
        results['total_metrics'] = self.analyze_total_revenue(df)
        
        # Product revenue analysis
        results['product_performance'] = self.analyze_revenue_by_product(df)
        
        # Daily revenue trends
        results['daily_revenue'] = self.analyze_daily_revenue(df)
        
        # Weekday revenue patterns
        results['weekday_revenue'] = self.analyze_weekday_revenue(df)
        
        # Monthly revenue trends
        results['monthly_revenue'] = self.analyze_monthly_revenue(df)
        
        # Price analysis
        results['price_by_coffee'] = self.analyze_price_by_product(df)
        
        self.results = results
        
        # Save outputs if requested
        if save_outputs:
            self.save_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Sales Performance Analysis Complete")
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
        
        self.logger.info("Saving sales performance results...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save DataFrames
        dataframe_outputs = {
            'product_performance': 'product_performance_detailed.csv',
            'daily_revenue': 'daily_revenue.csv',
            'price_by_coffee': 'price_by_coffee.csv'
        }
        
        for key, filename in dataframe_outputs.items():
            if key in results and isinstance(results[key], pd.DataFrame):
                filepath = output_dir / filename
                results[key].to_csv(filepath, index=False)
                saved_files[key] = filepath
                self.logger.info(f"Saved {key} to {filepath}")
        
        # Save summary metrics to JSON
        if 'total_metrics' in results:
            import json
            metrics_path = output_dir / 'sales_summary_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(results['total_metrics'], f, indent=2)
            saved_files['total_metrics'] = metrics_path
            self.logger.info(f"Saved total metrics to {metrics_path}")
        
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
        
        # Revenue insights
        if 'total_metrics' in results:
            metrics = results['total_metrics']
            insights.append(
                f"Total revenue generated: ${metrics['total_revenue']:,.2f} "
                f"from {metrics['total_transactions']:,} transactions"
            )
            insights.append(
                f"Average transaction value: ${metrics['average_transaction_value']:.2f} "
                f"(range: ${metrics['min_transaction']:.2f} - ${metrics['max_transaction']:.2f})"
            )
        
        # Product insights
        if 'product_performance' in results:
            products = results['product_performance']
            top_product = products.iloc[0]
            insights.append(
                f"Top revenue product: {top_product['coffee_name']} "
                f"contributing {top_product['revenue_percentage']:.1f}% of total revenue"
            )
            
            # 80/20 rule analysis
            top_20_pct = int(len(products) * 0.2)
            if top_20_pct > 0:
                revenue_from_top_20 = products.head(top_20_pct)['revenue_percentage'].sum()
                insights.append(
                    f"Top 20% of products ({top_20_pct} products) generate "
                    f"{revenue_from_top_20:.1f}% of revenue"
                )
        
        # Daily revenue insights
        if 'daily_revenue' in results:
            daily = results['daily_revenue']
            best_day = daily.loc[daily['total_revenue'].idxmax()]
            insights.append(
                f"Highest revenue day: {best_day['Date']} (${best_day['total_revenue']:,.2f})"
            )
        
        # Weekday insights
        if 'weekday_revenue' in results:
            weekday = results['weekday_revenue']
            best_weekday = weekday.loc[weekday['total_revenue'].idxmax()]
            insights.append(
                f"Best performing weekday: {best_weekday['Weekday']} "
                f"({best_weekday['revenue_percentage']:.1f}% of weekly revenue)"
            )
        
        return insights
