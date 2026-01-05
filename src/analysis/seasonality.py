"""Seasonality & Calendar Analysis Module (ARD Section 4.5).

Business Questions:
- Do sales fluctuate by month?
- Are there identifiable seasonal patterns?

Analysis Scope:
- Monthly revenue trends
- Month-over-month growth analysis
- Comparison using Monthsort
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class SeasonalityAnalyzer:
    """Analyzes seasonality and calendar patterns."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize SeasonalityAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.results: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def analyze_monthly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze revenue and transaction trends by month.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with monthly metrics
        """
        self.logger.info("Analyzing monthly trends...")
        
        if 'Month_name' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        monthly_metrics = df.groupby('Month_name').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        monthly_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate percentages
        monthly_metrics['revenue_percentage'] = (
            monthly_metrics['total_revenue'] / monthly_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        # Add month order if available
        if 'Monthsort' in df.columns:
            month_order = df.groupby('Month_name')['Monthsort'].first().to_dict()
            monthly_metrics['month_order'] = monthly_metrics.index.map(month_order)
            monthly_metrics = monthly_metrics.sort_values('month_order')
            
            # Calculate month-over-month growth
            monthly_metrics['revenue_mom_change'] = monthly_metrics['total_revenue'].diff().round(2)
            monthly_metrics['revenue_mom_growth_pct'] = (
                monthly_metrics['total_revenue'].pct_change() * 100
            ).round(2)
        
        monthly_metrics = monthly_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(monthly_metrics)} months")
        if not monthly_metrics.empty:
            best_month = monthly_metrics.loc[monthly_metrics['total_revenue'].idxmax()]
            self.logger.info(f"Best month: {best_month['Month_name']} (${best_month['total_revenue']:,.2f})")
        
        return monthly_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Identify seasonal patterns and high/low seasons.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with seasonal insights
        """
        self.logger.info("Analyzing seasonal patterns...")
        
        if 'Month_name' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        monthly_revenue = df.groupby('Month_name')['money'].sum()
        
        # Add month order for proper sorting
        if 'Monthsort' in df.columns:
            month_order = df.groupby('Month_name')['Monthsort'].first()
            monthly_revenue = monthly_revenue.to_frame()
            monthly_revenue['month_order'] = monthly_revenue.index.map(month_order)
            monthly_revenue = monthly_revenue.sort_values('month_order')['money']
        
        # Calculate statistics
        mean_revenue = monthly_revenue.mean()
        std_revenue = monthly_revenue.std()
        
        # Identify seasons
        high_season = monthly_revenue[monthly_revenue > mean_revenue + 0.5 * std_revenue]
        low_season = monthly_revenue[monthly_revenue < mean_revenue - 0.5 * std_revenue]
        
        seasonal_patterns = {
            'avg_monthly_revenue': float(mean_revenue),
            'std_monthly_revenue': float(std_revenue),
            'highest_month': {
                'month': str(monthly_revenue.idxmax()),
                'revenue': float(monthly_revenue.max())
            },
            'lowest_month': {
                'month': str(monthly_revenue.idxmin()),
                'revenue': float(monthly_revenue.min())
            },
            'high_season_months': high_season.index.tolist(),
            'low_season_months': low_season.index.tolist(),
            'revenue_range': float(monthly_revenue.max() - monthly_revenue.min()),
            'coefficient_of_variation': float((std_revenue / mean_revenue) * 100)
        }
        
        self.logger.info(f"High season months: {seasonal_patterns['high_season_months']}")
        self.logger.info(f"Low season months: {seasonal_patterns['low_season_months']}")
        
        return seasonal_patterns
    
    @log_execution_time(setup_logger(__name__))
    def analyze_quarter_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze performance by quarter (if date information available).
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with quarterly metrics
        """
        self.logger.info("Analyzing quarterly performance...")
        
        if 'Date' not in df.columns or 'money' not in df.columns:
            self.logger.warning("Required columns not found, skipping...")
            return pd.DataFrame()
        
        # Extract quarter
        df_temp = df.copy()
        df_temp['quarter'] = pd.to_datetime(df_temp['Date']).dt.quarter
        df_temp['year'] = pd.to_datetime(df_temp['Date']).dt.year
        
        quarterly_metrics = df_temp.groupby(['year', 'quarter']).agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        quarterly_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Calculate quarter-over-quarter growth
        quarterly_metrics['revenue_qoq_change'] = quarterly_metrics['total_revenue'].diff().round(2)
        quarterly_metrics['revenue_qoq_growth_pct'] = (
            quarterly_metrics['total_revenue'].pct_change() * 100
        ).round(2)
        
        quarterly_metrics = quarterly_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(quarterly_metrics)} quarters")
        
        return quarterly_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_day_of_month_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze patterns by day of month (e.g., month start/end effects).
        
        Args:
            df: Sales DataFrame
            
        Returns:
            DataFrame with day-of-month metrics
        """
        self.logger.info("Analyzing day of month patterns...")
        
        if 'Date' not in df.columns or 'money' not in df.columns:
            self.logger.warning("Required columns not found, skipping...")
            return pd.DataFrame()
        
        df_temp = df.copy()
        df_temp['day_of_month'] = pd.to_datetime(df_temp['Date']).dt.day
        
        day_metrics = df_temp.groupby('day_of_month').agg({
            'money': ['sum', 'mean', 'count']
        }).round(2)
        
        day_metrics.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
        
        # Identify period (start, mid, end of month)
        day_metrics['period'] = pd.cut(
            day_metrics.index,
            bins=[0, 10, 20, 31],
            labels=['Month Start', 'Mid Month', 'Month End'],
            include_lowest=True
        )
        
        day_metrics = day_metrics.reset_index()
        
        self.logger.info(f"Analyzed {len(day_metrics)} days of month")
        
        return day_metrics
    
    @log_execution_time(setup_logger(__name__))
    def analyze_growth_trends(self, df: pd.DataFrame) -> Dict:
        """
        Calculate overall growth trends and forecasting indicators.
        
        Args:
            df: Sales DataFrame
            
        Returns:
            Dictionary with growth metrics
        """
        self.logger.info("Analyzing growth trends...")
        
        if 'Date' not in df.columns or 'money' not in df.columns:
            raise ValueError("Required columns not found")
        
        # Daily revenue
        daily_revenue = df.groupby('Date')['money'].sum().sort_index()
        
        # Calculate overall trend
        from scipy import stats
        x = np.arange(len(daily_revenue))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_revenue.values)
        
        growth_metrics = {
            'total_days': int(len(daily_revenue)),
            'start_date': str(daily_revenue.index.min()),
            'end_date': str(daily_revenue.index.max()),
            'first_day_revenue': float(daily_revenue.iloc[0]),
            'last_day_revenue': float(daily_revenue.iloc[-1]),
            'avg_daily_revenue': float(daily_revenue.mean()),
            'trend_slope': float(slope),
            'trend_r_squared': float(r_value ** 2),
            'overall_growth_pct': float(((daily_revenue.iloc[-1] - daily_revenue.iloc[0]) / daily_revenue.iloc[0]) * 100),
            'avg_daily_growth': float(slope)
        }
        
        trend_direction = "increasing" if slope > 0 else "decreasing"
        self.logger.info(f"Revenue trend: {trend_direction} (R²={growth_metrics['trend_r_squared']:.3f})")
        
        return growth_metrics
    
    @log_execution_time(setup_logger(__name__))
    def run_analysis(self, df: pd.DataFrame, save_outputs: bool = True) -> Dict:
        """
        Execute complete seasonality analysis.
        
        Args:
            df: Sales DataFrame
            save_outputs: Whether to save outputs to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("SEASONALITY & CALENDAR ANALYSIS (ARD Section 4.5)")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Monthly trends
        results['monthly_trends'] = self.analyze_monthly_trends(df)
        
        # Seasonal patterns
        results['seasonal_patterns'] = self.analyze_seasonal_patterns(df)
        
        # Quarterly performance
        results['quarterly_performance'] = self.analyze_quarter_performance(df)
        
        # Day of month patterns
        results['day_of_month_patterns'] = self.analyze_day_of_month_patterns(df)
        
        # Growth trends
        results['growth_trends'] = self.analyze_growth_trends(df)
        
        self.results = results
        
        # Save outputs if requested
        if save_outputs:
            self.save_results(results)
        
        self.logger.info("=" * 80)
        self.logger.info("Seasonality & Calendar Analysis Complete")
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
        
        self.logger.info("Saving seasonality analysis results...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save DataFrames
        dataframe_outputs = {
            'monthly_trends': 'monthly_trends_analysis.csv',
            'quarterly_performance': 'quarterly_performance.csv',
            'day_of_month_patterns': 'day_of_month_patterns.csv'
        }
        
        for key, filename in dataframe_outputs.items():
            if key in results and isinstance(results[key], pd.DataFrame) and not results[key].empty:
                filepath = output_dir / filename
                results[key].to_csv(filepath, index=False)
                saved_files[key] = filepath
                self.logger.info(f"Saved {key} to {filepath}")
        
        # Save dictionaries to JSON
        import json
        json_outputs = {
            'seasonal_patterns': 'seasonal_patterns.json',
            'growth_trends': 'growth_trends.json'
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
        
        # Seasonal insights
        if 'seasonal_patterns' in results:
            patterns = results['seasonal_patterns']
            insights.append(
                f"Peak sales month: {patterns['highest_month']['month']} "
                f"(${patterns['highest_month']['revenue']:,.2f})"
            )
            insights.append(
                f"Lowest sales month: {patterns['lowest_month']['month']} "
                f"(${patterns['lowest_month']['revenue']:,.2f})"
            )
            
            if patterns['high_season_months']:
                insights.append(
                    f"High season: {', '.join(patterns['high_season_months'])}"
                )
        
        # Growth insights
        if 'growth_trends' in results:
            growth = results['growth_trends']
            trend = "growing" if growth['trend_slope'] > 0 else "declining"
            insights.append(
                f"Revenue trend: {trend} at ${abs(growth['avg_daily_growth']):.2f}/day "
                f"(R²={growth['trend_r_squared']:.3f})"
            )
        
        # Monthly variation insights
        if 'seasonal_patterns' in results:
            patterns = results['seasonal_patterns']
            cv = patterns['coefficient_of_variation']
            if cv < 10:
                variation = "low"
            elif cv < 20:
                variation = "moderate"
            else:
                variation = "high"
            insights.append(
                f"Monthly revenue variation: {variation} (CV={cv:.1f}%)"
            )
        
        return insights
