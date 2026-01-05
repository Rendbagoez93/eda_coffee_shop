"""Time-Based Demand Analysis Module (ARD Section 4.2).

Business Questions:
- What are the peak and off-peak hours?
- How does demand differ by time of day?
- Are there weekday vs weekend differences?

Analysis Scope:
- Sales and revenue by hour_of_day
- Sales by Time_of_Day
- Sales by Weekday
- Hour vs Weekday demand heatmap data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class TimeDemandAnalyzer:
    """Analyzes time-based demand patterns."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize TimeDemandAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.results: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def analyze_hourly_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze demand patterns by hour of day.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with hourly demand metrics
        """
        self.logger.info("Analyzing hourly demand patterns...")
        
        if 'hour_of_day' not in df.columns:
            raise ValueError("'hour_of_day' column not found")
        
        hourly_metrics = df.groupby('hour_of_day').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        hourly_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentages
        hourly_metrics['revenue_percentage'] = (
            hourly_metrics['total_revenue'] / hourly_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        hourly_metrics['transaction_percentage'] = (
            hourly_metrics['transaction_count'] / hourly_metrics['transaction_count'].sum() * 100
        ).round(2)
        
        # Identify peak hours (top 20% by transaction count)
        threshold = hourly_metrics['transaction_count'].quantile(0.80)
        hourly_metrics['is_peak_hour'] = (hourly_metrics['transaction_count'] >= threshold).astype(int)
        
        # Add time segments
        hourly_metrics['time_segment'] = pd.cut(
            hourly_metrics.index,
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        hourly_metrics = hourly_metrics.reset_index()
        
        peak_hours = hourly_metrics[hourly_metrics['is_peak_hour'] == 1]['hour_of_day'].tolist()
        self.logger.info(f"Peak hours identified: {peak_hours}")
        
        return hourly_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_time_of_day_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze demand by time of day segments.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with time-of-day metrics
        """
        self.logger.info("Analyzing demand by time of day segments...")
        
        if 'Time_of_Day' not in df.columns:
            self.logger.warning("'Time_of_Day' column not found, skipping...")
            return pd.DataFrame()
        
        time_metrics = df.groupby('Time_of_Day').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        time_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentages
        time_metrics['revenue_percentage'] = (
            time_metrics['total_revenue'] / time_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        time_metrics['transaction_percentage'] = (
            time_metrics['transaction_count'] / time_metrics['transaction_count'].sum() * 100
        ).round(2)
        
        time_metrics = time_metrics.reset_index()
        time_metrics = time_metrics.sort_values('total_revenue', ascending=False)
        
        self.logger.info(f"Analyzed {len(time_metrics)} time segments")
        
        return time_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_weekday_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze demand patterns by weekday.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with weekday demand metrics
        """
        self.logger.info("Analyzing weekday demand patterns...")
        
        if 'Weekday' not in df.columns:
            raise ValueError("'Weekday' column not found")
        
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekday_metrics = df.groupby('Weekday').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        weekday_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentages
        weekday_metrics['revenue_percentage'] = (
            weekday_metrics['total_revenue'] / weekday_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        weekday_metrics['transaction_percentage'] = (
            weekday_metrics['transaction_count'] / weekday_metrics['transaction_count'].sum() * 100
        ).round(2)
        
        # Add weekend indicator
        weekday_metrics['is_weekend'] = weekday_metrics.index.isin(['Saturday', 'Sunday']).astype(int)
        
        weekday_metrics = weekday_metrics.reset_index()
        
        # Sort by weekday order
        weekday_metrics['day_order'] = weekday_metrics['Weekday'].map({day: i for i, day in enumerate(weekday_order)})
        weekday_metrics = weekday_metrics.sort_values('day_order').drop('day_order', axis=1)
        
        return weekday_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_hour_weekday_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create heatmap data for hour vs weekday analysis.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Pivot table DataFrame for heatmap
        """
        self.logger.info("Creating hour-weekday heatmap data...")
        
        if 'hour_of_day' not in df.columns or 'Weekday' not in df.columns:
            raise ValueError("Required columns not found")
        
        # Transaction count heatmap
        heatmap_data = df.groupby(['Weekday', 'hour_of_day']).size().unstack(fill_value=0)
        
        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex([day for day in weekday_order if day in heatmap_data.index])
        
        self.logger.info(f"Heatmap data created: {heatmap_data.shape}")
        
        return heatmap_data
    
    @log_execution_time(setup_logger(__name__))
    def analyze_peak_vs_offpeak(self, df: pd.DataFrame) -> Dict:
        """
        Compare peak vs off-peak period metrics.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with peak/off-peak comparison
        """
        self.logger.info("Analyzing peak vs off-peak periods...")
        
        if 'hour_of_day' not in df.columns:
            raise ValueError("'hour_of_day' column not found")
        
        # Define peak hours (typically 7-10 AM and 12-3 PM for coffee shops)
        peak_hours = [7, 8, 9, 10, 12, 13, 14, 15]
        df_temp = df.copy()
        df_temp['period'] = df_temp['hour_of_day'].apply(
            lambda h: 'Peak' if h in peak_hours else 'Off-Peak'
        )
        
        period_metrics = df_temp.groupby('period').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        period_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentages
        period_metrics['revenue_percentage'] = (
            period_metrics['total_revenue'] / period_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        comparison = period_metrics.to_dict('index')
        
        self.logger.info(f"Peak period revenue: ${comparison.get('Peak', {}).get('total_revenue', 0):,.2f}")
        self.logger.info(f"Off-Peak period revenue: ${comparison.get('Off-Peak', {}).get('total_revenue', 0):,.2f}")
        
        return comparison
    
    @log_execution_time(setup_logger(__name__))
    def analyze_weekend_vs_weekday(self, df: pd.DataFrame) -> Dict:
        """
        Compare weekend vs weekday demand patterns.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with weekend/weekday comparison
        """
        self.logger.info("Analyzing weekend vs weekday patterns...")
        
        if 'Weekday' not in df.columns:
            raise ValueError("'Weekday' column not found")
        
        df_temp = df.copy()
        df_temp['day_type'] = df_temp['Weekday'].apply(
            lambda d: 'Weekend' if d in ['Saturday', 'Sunday'] else 'Weekday'
        )
        
        daytype_metrics = df_temp.groupby('day_type').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        daytype_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate per-day averages
        daytype_metrics['avg_daily_revenue'] = daytype_metrics['total_revenue'] / daytype_metrics.apply(
            lambda row: 2 if row.name == 'Weekend' else 5, axis=1
        )
        
        daytype_metrics['avg_daily_transactions'] = daytype_metrics['transaction_count'] / daytype_metrics.apply(
            lambda row: 2 if row.name == 'Weekend' else 5, axis=1
        )
        
        comparison = daytype_metrics.to_dict('index')
        
        self.logger.info(f"Weekday avg daily revenue: ${comparison.get('Weekday', {}).get('avg_daily_revenue', 0):,.2f}")
        self.logger.info(f"Weekend avg daily revenue: ${comparison.get('Weekend', {}).get('avg_daily_revenue', 0):,.2f}")
        
        return comparison
    
    @log_execution_time(setup_logger(__name__))
    def run_analysis(self, df: pd.DataFrame, save_outputs: bool = True) -> Dict:
        """
        Execute complete time-based demand analysis.
        
        Args:
            df: Sales DataFrame
            save_outputs: Whether to save outputs to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("TIME-BASED DEMAND ANALYSIS (ARD Section 4.2)")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Hourly demand analysis
        results['hourly_demand'] = self.analyze_hourly_demand(df)
        
        # Time of day analysis
        results['time_of_day_demand'] = self.analyze_time_of_day_demand(df)
        
        # Weekday patterns
        results['weekday_patterns'] = self.analyze_weekday_patterns(df)
        
        # Hour-weekday heatmap
        results['hour_weekday_heatmap'] = self.analyze_hour_weekday_heatmap(df)
        
        # Peak vs off-peak comparison
        results['peak_offpeak_comparison'] = self.analyze_peak_vs_offpeak(df)
        
        # Weekend vs weekday comparison
        results['weekend_weekday_comparison'] = self.analyze_weekend_vs_weekday(df)
        
        self.results = results
        
        # Save outputs if requested
        if save_outputs:
            self.save_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Time-Based Demand Analysis Complete")
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
        
        self.logger.info("Saving time demand analysis results...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save DataFrames
        dataframe_outputs = {
            'hourly_demand': 'hourly_detailed_analysis.csv',
            'hour_weekday_heatmap': 'hour_weekday_heatmap.csv'
        }
        
        for key, filename in dataframe_outputs.items():
            if key in results and isinstance(results[key], pd.DataFrame):
                filepath = output_dir / filename
                results[key].to_csv(filepath, index=False if key != 'hour_weekday_heatmap' else True)
                saved_files[key] = filepath
                self.logger.info(f"Saved {key} to {filepath}")
        
        # Save comparisons to JSON
        import json
        json_outputs = {
            'peak_offpeak_comparison': 'peak_offpeak_comparison.json',
            'weekend_weekday_comparison': 'weekend_weekday_comparison.json'
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
        
        # Hourly insights
        if 'hourly_demand' in results:
            hourly = results['hourly_demand']
            peak_hours = hourly[hourly['is_peak_hour'] == 1].sort_values('transaction_count', ascending=False)
            if not peak_hours.empty:
                top_hour = peak_hours.iloc[0]
                insights.append(
                    f"Peak demand hour: {int(top_hour['hour_of_day'])}:00 "
                    f"({int(top_hour['transaction_count'])} transactions, "
                    f"{top_hour['transaction_percentage']:.1f}% of daily volume)"
                )
        
        # Peak vs off-peak insights
        if 'peak_offpeak_comparison' in results:
            comp = results['peak_offpeak_comparison']
            if 'Peak' in comp and 'Off-Peak' in comp:
                peak_pct = comp['Peak'].get('revenue_percentage', 0)
                insights.append(
                    f"Peak hours generate {peak_pct:.1f}% of daily revenue despite being a fraction of operating hours"
                )
        
        # Weekend vs weekday insights
        if 'weekend_weekday_comparison' in results:
            comp = results['weekend_weekday_comparison']
            if 'Weekend' in comp and 'Weekday' in comp:
                weekend_avg = comp['Weekend'].get('avg_daily_revenue', 0)
                weekday_avg = comp['Weekday'].get('avg_daily_revenue', 0)
                
                if weekend_avg > weekday_avg:
                    diff_pct = ((weekend_avg - weekday_avg) / weekday_avg * 100)
                    insights.append(
                        f"Weekend days generate {diff_pct:.1f}% more revenue on average than weekdays"
                    )
                else:
                    diff_pct = ((weekday_avg - weekend_avg) / weekend_avg * 100)
                    insights.append(
                        f"Weekdays generate {diff_pct:.1f}% more revenue on average than weekends"
                    )
        
        return insights
