"""Payment Behavior Analysis Module (ARD Section 4.4).

Business Questions:
- What payment methods do customers prefer?
- Does payment type influence spending value?
- Does payment behavior vary by time?

Analysis Scope:
- Transaction count by cash_type
- Revenue contribution by payment method
- Payment method usage by hour and weekday
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class PaymentBehaviorAnalyzer:
    """Analyzes payment behavior and patterns."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize PaymentBehaviorAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.results: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def analyze_payment_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze distribution of payment methods.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with payment method metrics
        """
        self.logger.info("Analyzing payment method distribution...")
        
        if 'cash_type' not in df.columns:
            raise ValueError("'cash_type' column not found")
        
        payment_metrics = df.groupby('cash_type').agg({
            'money': ['sum', 'mean', 'median', 'count']
        }).round(2)
        
        payment_metrics.columns = ['total_revenue', 'avg_transaction', 'median_transaction', 'transaction_count']
        
        # Calculate percentages
        payment_metrics['revenue_percentage'] = (
            payment_metrics['total_revenue'] / payment_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        payment_metrics['transaction_percentage'] = (
            payment_metrics['transaction_count'] / payment_metrics['transaction_count'].sum() * 100
        ).round(2)
        
        payment_metrics = payment_metrics.reset_index()
        payment_metrics = payment_metrics.sort_values('transaction_count', ascending=False)
        
        self.logger.info(f"Analyzed {len(payment_metrics)} payment methods")
        if not payment_metrics.empty:
            top_method = payment_metrics.iloc[0]
            self.logger.info(f"Most used payment: {top_method['cash_type']} ({top_method['transaction_percentage']:.1f}%)")
        
        return payment_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_spending_by_payment(self, df: pd.DataFrame) -> Dict:
        """
        Compare spending patterns across payment methods.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with spending comparison metrics
        """
        self.logger.info("Analyzing spending patterns by payment method...")
        
        if 'cash_type' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        spending_stats = {}
        
        for payment_type in df['cash_type'].unique():
            payment_data = df[df['cash_type'] == payment_type]['money']
            
            spending_stats[str(payment_type)] = {
                'avg_spend': float(payment_data.mean()),
                'median_spend': float(payment_data.median()),
                'std_spend': float(payment_data.std()),
                'min_spend': float(payment_data.min()),
                'max_spend': float(payment_data.max()),
                'total_transactions': int(len(payment_data))
            }
        
        # Calculate spend difference
        if len(spending_stats) == 2:
            methods = list(spending_stats.keys())
            avg_diff = abs(spending_stats[methods[0]]['avg_spend'] - spending_stats[methods[1]]['avg_spend'])
            higher_method = methods[0] if spending_stats[methods[0]]['avg_spend'] > spending_stats[methods[1]]['avg_spend'] else methods[1]
            
            self.logger.info(f"{higher_method} has ${avg_diff:.2f} higher average spend")
        
        return spending_stats
    
    @log_execution_time(setup_logger(__name__))
    def analyze_payment_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze payment method preferences by hour.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with hourly payment patterns
        """
        self.logger.info("Analyzing payment methods by hour...")
        
        if 'cash_type' not in df.columns or 'hour_of_day' not in df.columns:
            raise ValueError("Required columns not found")
        
        payment_hour = df.groupby(['hour_of_day', 'cash_type']).size().reset_index(name='transaction_count')
        
        # Calculate percentage by hour
        hourly_totals = payment_hour.groupby('hour_of_day')['transaction_count'].transform('sum')
        payment_hour['pct_of_hour'] = (payment_hour['transaction_count'] / hourly_totals * 100).round(2)
        
        # Create pivot for better visualization
        payment_hour_pivot = payment_hour.pivot(
            index='hour_of_day',
            columns='cash_type',
            values='pct_of_hour'
        ).fillna(0).round(2)
        
        self.logger.info(f"Analyzed payment patterns across {len(payment_hour_pivot)} hours")
        
        return payment_hour_pivot
    
    @log_execution_time(setup_logger(__name__))
    def analyze_payment_by_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze payment method preferences by weekday.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with weekday payment patterns
        """
        self.logger.info("Analyzing payment methods by weekday...")
        
        if 'cash_type' not in df.columns or 'Weekday' not in df.columns:
            raise ValueError("Required columns not found")
        
        payment_weekday = df.groupby(['Weekday', 'cash_type']).size().reset_index(name='transaction_count')
        
        # Calculate percentage by weekday
        weekday_totals = payment_weekday.groupby('Weekday')['transaction_count'].transform('sum')
        payment_weekday['pct_of_day'] = (payment_weekday['transaction_count'] / weekday_totals * 100).round(2)
        
        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        payment_weekday['day_order'] = payment_weekday['Weekday'].map({day: i for i, day in enumerate(weekday_order)})
        payment_weekday = payment_weekday.sort_values('day_order').drop('day_order', axis=1)
        
        self.logger.info(f"Analyzed payment patterns across {payment_weekday['Weekday'].nunique()} weekdays")
        
        return payment_weekday
    
    @log_execution_time(setup_logger(__name__))
    def analyze_payment_by_time_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze payment preferences by time of day segment.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with time segment payment patterns
        """
        self.logger.info("Analyzing payment methods by time segment...")
        
        if 'cash_type' not in df.columns or 'Time_of_Day' not in df.columns:
            self.logger.warning("Required columns not found, skipping...")
            return pd.DataFrame()
        
        payment_time = df.groupby(['Time_of_Day', 'cash_type']).agg({
            'money': ['sum', 'count']
        }).round(2)
        
        payment_time.columns = ['total_revenue', 'transaction_count']
        
        # Calculate percentages
        time_totals = payment_time.groupby(level=0)['transaction_count'].transform('sum')
        payment_time['pct_of_period'] = (payment_time['transaction_count'] / time_totals * 100).round(2)
        
        payment_time = payment_time.reset_index()
        
        self.logger.info(f"Analyzed {len(payment_time)} time-payment combinations")
        
        return payment_time
    
    @log_execution_time(setup_logger(__name__))
    def analyze_cash_vs_cashless_trends(self, df: pd.DataFrame) -> Dict:
        """
        Compare cash vs cashless payment trends.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with cash/cashless comparison
        """
        self.logger.info("Analyzing cash vs cashless trends...")
        
        if 'cash_type' not in df.columns:
            raise ValueError("'cash_type' column not found")
        
        # Normalize payment types to cash/cashless
        df_temp = df.copy()
        df_temp['payment_category'] = df_temp['cash_type'].str.lower().apply(
            lambda x: 'Cash' if 'cash' in str(x) else 'Cashless'
        )
        
        category_metrics = df_temp.groupby('payment_category').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        category_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate adoption rates
        category_metrics['transaction_percentage'] = (
            category_metrics['transaction_count'] / category_metrics['transaction_count'].sum() * 100
        ).round(2)
        
        category_metrics['revenue_percentage'] = (
            category_metrics['total_revenue'] / category_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        comparison = category_metrics.to_dict('index')
        
        if 'Cashless' in comparison:
            self.logger.info(f"Cashless adoption rate: {comparison['Cashless']['transaction_percentage']:.1f}%")
        
        return comparison
    
    @log_execution_time(setup_logger(__name__))
    def run_analysis(self, df: pd.DataFrame, save_outputs: bool = True) -> Dict:
        """
        Execute complete payment behavior analysis.
        
        Args:
            df: Sales DataFrame
            save_outputs: Whether to save outputs to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("PAYMENT BEHAVIOR ANALYSIS (ARD Section 4.4)")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Payment distribution
        results['payment_distribution'] = self.analyze_payment_distribution(df)
        
        # Spending by payment
        results['spending_by_payment'] = self.analyze_spending_by_payment(df)
        
        # Payment by hour
        results['payment_by_hour'] = self.analyze_payment_by_hour(df)
        
        # Payment by weekday
        results['payment_by_weekday'] = self.analyze_payment_by_weekday(df)
        
        # Payment by time segment
        results['payment_by_time_segment'] = self.analyze_payment_by_time_segment(df)
        
        # Cash vs cashless trends
        results['cash_cashless_comparison'] = self.analyze_cash_vs_cashless_trends(df)
        
        self.results = results
        
        # Save outputs if requested
        if save_outputs:
            self.save_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Payment Behavior Analysis Complete")
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
        
        self.logger.info("Saving payment behavior analysis results...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save DataFrames
        dataframe_outputs = {
            'payment_distribution': 'payment_distribution.csv',
            'payment_by_hour': 'payment_hourly_patterns.csv',
            'payment_by_weekday': 'payment_weekday_patterns.csv',
            'payment_by_time_segment': 'payment_time_segment_analysis.csv'
        }
        
        for key, filename in dataframe_outputs.items():
            if key in results and isinstance(results[key], pd.DataFrame) and not results[key].empty:
                filepath = output_dir / filename
                if key == 'payment_by_hour':
                    results[key].to_csv(filepath, index=True)
                else:
                    results[key].to_csv(filepath, index=False)
                saved_files[key] = filepath
                self.logger.info(f"Saved {key} to {filepath}")
        
        # Save dictionaries to JSON
        import json
        json_outputs = {
            'spending_by_payment': 'spending_by_payment.json',
            'cash_cashless_comparison': 'cash_cashless_comparison.json'
        }
        
        for key, filename in json_outputs.items():
            if key in results:
                filepath = output_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(results[key], f, indent=2)
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
        
        # Payment preference insights
        if 'payment_distribution' in results and not results['payment_distribution'].empty:
            dist = results['payment_distribution']
            top_payment = dist.iloc[0]
            insights.append(
                f"Preferred payment method: {top_payment['cash_type']} "
                f"({top_payment['transaction_percentage']:.1f}% of transactions)"
            )
        
        # Spending difference insights
        if 'spending_by_payment' in results:
            spending = results['spending_by_payment']
            if len(spending) >= 2:
                methods = list(spending.keys())
                avg1, avg2 = spending[methods[0]]['avg_spend'], spending[methods[1]]['avg_spend']
                higher_method = methods[0] if avg1 > avg2 else methods[1]
                diff_pct = abs((avg1 - avg2) / min(avg1, avg2) * 100)
                insights.append(
                    f"{higher_method} transactions are {diff_pct:.1f}% higher in average value"
                )
        
        # Cash/cashless insights
        if 'cash_cashless_comparison' in results:
            comp = results['cash_cashless_comparison']
            if 'Cashless' in comp:
                cashless_pct = comp['Cashless']['transaction_percentage']
                insights.append(
                    f"Cashless payment adoption: {cashless_pct:.1f}% of all transactions"
                )
        
        return insights
