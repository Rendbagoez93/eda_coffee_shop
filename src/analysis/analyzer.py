"""Main Analysis Orchestrator for Coffee Sales Analysis.

This module coordinates all analysis modules and provides
a unified interface for running the complete analysis pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

from .sales_performance import SalesPerformanceAnalyzer
from .time_demand import TimeDemandAnalyzer
from .product_preference import ProductPreferenceAnalyzer
from .payment_behavior import PaymentBehaviorAnalyzer
from .seasonality import SeasonalityAnalyzer

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_execution_time


class CoffeeSalesAnalyzer:
    """Main orchestrator for complete coffee sales analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize CoffeeSalesAnalyzer.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        # Initialize all analyzers
        self.sales_analyzer = SalesPerformanceAnalyzer(config_dir)
        self.time_analyzer = TimeDemandAnalyzer(config_dir)
        self.product_analyzer = ProductPreferenceAnalyzer(config_dir)
        self.payment_analyzer = PaymentBehaviorAnalyzer(config_dir)
        self.seasonality_analyzer = SeasonalityAnalyzer(config_dir)
        
        self.all_results: Dict = {}
        self.all_insights: List[str] = []
    
    @log_execution_time(setup_logger(__name__))
    def run_all_analyses(
        self,
        df: pd.DataFrame,
        save_outputs: bool = True,
        analyses: Optional[List[str]] = None
    ) -> Dict:
        """
        Execute all configured analyses.
        
        Args:
            df: Enriched sales DataFrame
            save_outputs: Whether to save outputs to files
            analyses: List of analyses to run. If None, runs all enabled analyses.
                     Options: ['sales', 'time', 'product', 'payment', 'seasonality']
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE COFFEE SALES ANALYSIS")
        self.logger.info("Based on ARD - Analysis Requirements Document")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Determine which analyses to run
        if analyses is None:
            analyses = ['sales', 'time', 'product', 'payment', 'seasonality']
        
        # Section 4.1: Sales Performance Analysis
        if 'sales' in analyses:
            if self.config.get('analysis', {}).get('sales_performance', {}).get('enabled', True):
                self.logger.info("\n" + "=" * 80)
                results['sales_performance'] = self.sales_analyzer.run_analysis(df, save_outputs)
            else:
                self.logger.info("Sales Performance Analysis disabled in config")
        
        # Section 4.2: Time-Based Demand Analysis
        if 'time' in analyses:
            if self.config.get('analysis', {}).get('time_based_demand', {}).get('enabled', True):
                self.logger.info("\n" + "=" * 80)
                results['time_demand'] = self.time_analyzer.run_analysis(df, save_outputs)
            else:
                self.logger.info("Time-Based Demand Analysis disabled in config")
        
        # Section 4.3: Product Preference Analysis
        if 'product' in analyses:
            if self.config.get('analysis', {}).get('product_preference', {}).get('enabled', True):
                self.logger.info("\n" + "=" * 80)
                results['product_preference'] = self.product_analyzer.run_analysis(df, save_outputs)
            else:
                self.logger.info("Product Preference Analysis disabled in config")
        
        # Section 4.4: Payment Behavior Analysis
        if 'payment' in analyses:
            if self.config.get('analysis', {}).get('payment_behavior', {}).get('enabled', True):
                self.logger.info("\n" + "=" * 80)
                results['payment_behavior'] = self.payment_analyzer.run_analysis(df, save_outputs)
            else:
                self.logger.info("Payment Behavior Analysis disabled in config")
        
        # Section 4.5: Seasonality & Calendar Analysis
        if 'seasonality' in analyses:
            if self.config.get('analysis', {}).get('seasonality', {}).get('enabled', True):
                self.logger.info("\n" + "=" * 80)
                results['seasonality'] = self.seasonality_analyzer.run_analysis(df, save_outputs)
            else:
                self.logger.info("Seasonality Analysis disabled in config")
        
        self.all_results = results
        
        # Generate consolidated insights
        self.generate_insights()
        
        # Save executive summary
        if save_outputs:
            self.save_executive_summary()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ALL ANALYSES COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        
        return results
    
    def generate_insights(self) -> List[str]:
        """
        Generate consolidated business insights from all analyses.
        
        Returns:
            List of key business insights
        """
        self.logger.info("Generating consolidated business insights...")
        
        all_insights = []
        
        # Get insights from each analyzer
        if hasattr(self.sales_analyzer, 'results') and self.sales_analyzer.results:
            sales_insights = self.sales_analyzer.get_insights()
            all_insights.extend([f"[Sales] {i}" for i in sales_insights])
        
        if hasattr(self.time_analyzer, 'results') and self.time_analyzer.results:
            time_insights = self.time_analyzer.get_insights()
            all_insights.extend([f"[Time] {i}" for i in time_insights])
        
        if hasattr(self.product_analyzer, 'results') and self.product_analyzer.results:
            product_insights = self.product_analyzer.get_insights()
            all_insights.extend([f"[Product] {i}" for i in product_insights])
        
        if hasattr(self.payment_analyzer, 'results') and self.payment_analyzer.results:
            payment_insights = self.payment_analyzer.get_insights()
            all_insights.extend([f"[Payment] {i}" for i in payment_insights])
        
        if hasattr(self.seasonality_analyzer, 'results') and self.seasonality_analyzer.results:
            seasonality_insights = self.seasonality_analyzer.get_insights()
            all_insights.extend([f"[Seasonality] {i}" for i in seasonality_insights])
        
        self.all_insights = all_insights
        
        self.logger.info(f"Generated {len(all_insights)} key insights")
        
        return all_insights
    
    def save_executive_summary(self) -> Path:
        """
        Save executive summary with key findings and recommendations.
        
        Returns:
            Path to saved summary file
        """
        self.logger.info("Generating executive summary...")
        
        output_dir = Path(self.paths.get('outputs', {}).get('root', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / 'executive_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COFFEE SALES ANALYSIS - EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Project: {self.config.get('project', {}).get('name', 'Coffee Sales Analysis')}\n")
            f.write(f"Version: {self.config.get('project', {}).get('version', '1.0.0')}\n")
            f.write(f"Date: {self.config.get('project', {}).get('date_created', 'N/A')}\n\n")
            
            # Objectives
            f.write("BUSINESS OBJECTIVES\n")
            f.write("-" * 80 + "\n")
            f.write("1. Revenue Optimization\n")
            f.write("2. Operational Efficiency\n")
            f.write("3. Product & Marketing Strategy\n\n")
            
            # Key Insights
            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n")
            if self.all_insights:
                for i, insight in enumerate(self.all_insights, 1):
                    f.write(f"{i}. {insight}\n")
            else:
                f.write("No insights generated yet.\n")
            f.write("\n")
            
            # Analysis Summary
            f.write("ANALYSES COMPLETED\n")
            f.write("-" * 80 + "\n")
            analyses_completed = []
            if 'sales_performance' in self.all_results:
                analyses_completed.append("✓ Sales Performance Analysis (ARD 4.1)")
            if 'time_demand' in self.all_results:
                analyses_completed.append("✓ Time-Based Demand Analysis (ARD 4.2)")
            if 'product_preference' in self.all_results:
                analyses_completed.append("✓ Product Preference Analysis (ARD 4.3)")
            if 'payment_behavior' in self.all_results:
                analyses_completed.append("✓ Payment Behavior Analysis (ARD 4.4)")
            if 'seasonality' in self.all_results:
                analyses_completed.append("✓ Seasonality & Calendar Analysis (ARD 4.5)")
            
            for analysis in analyses_completed:
                f.write(f"{analysis}\n")
            f.write("\n")
            
            # Recommendations
            f.write("ACTIONABLE RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("1. Optimize staffing based on identified peak hours\n")
            f.write("2. Adjust inventory for top-performing products\n")
            f.write("3. Implement time-based promotions during off-peak periods\n")
            f.write("4. Focus marketing on high-revenue products and time slots\n")
            f.write("5. Plan for seasonal variations in demand\n\n")
            
            # Output Files
            f.write("OUTPUT FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write("All analysis results have been saved to the 'outputs' directory.\n")
            f.write("See individual CSV and JSON files for detailed metrics.\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Executive Summary\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Executive summary saved to {summary_path}")
        
        # Also save as JSON for programmatic access
        json_summary = {
            'project': self.config.get('project', {}),
            'insights': self.all_insights,
            'analyses_completed': analyses_completed,
            'timestamp': str(pd.Timestamp.now())
        }
        
        json_path = output_dir / 'executive_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2)
        
        self.logger.info(f"Executive summary JSON saved to {json_path}")
        
        return summary_path
    
    def get_summary_statistics(self) -> Dict:
        """
        Get high-level summary statistics across all analyses.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'analyses_run': list(self.all_results.keys()),
            'total_insights': len(self.all_insights),
            'key_metrics': {}
        }
        
        # Extract key metrics from each analysis
        if 'sales_performance' in self.all_results:
            sp = self.all_results['sales_performance']
            if 'total_metrics' in sp:
                summary['key_metrics']['total_revenue'] = sp['total_metrics'].get('total_revenue')
                summary['key_metrics']['total_transactions'] = sp['total_metrics'].get('total_transactions')
                summary['key_metrics']['avg_transaction_value'] = sp['total_metrics'].get('average_transaction_value')
        
        return summary
    
    def print_insights(self):
        """Print all insights to console."""
        if not self.all_insights:
            self.logger.warning("No insights available. Run analyses first.")
            return
        
        print("\n" + "=" * 80)
        print("KEY BUSINESS INSIGHTS")
        print("=" * 80)
        for i, insight in enumerate(self.all_insights, 1):
            print(f"{i}. {insight}")
        print("=" * 80 + "\n")
