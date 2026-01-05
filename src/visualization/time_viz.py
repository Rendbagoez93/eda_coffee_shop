"""Time-Based Demand Visualizations (ARD Section 4.2)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_viz import BaseVisualizer
from ..utils.logger import log_execution_time, setup_logger


class TimeVisualizer(BaseVisualizer):
    """Create visualizations for time-based demand analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize TimeVisualizer."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
    
    @log_execution_time(setup_logger(__name__))
    def plot_hourly_demand(self) -> Path:
        """
        Create hourly demand visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating hourly demand visualization...")
        
        # Load data
        hourly_data = self.load_csv_data('hourly_detailed_analysis.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 1, figsize=(16, 10))
        
        # 1. Revenue and Transactions by Hour
        ax = axes[0, 0]
        
        ax2 = ax.twinx()
        
        # Plot revenue as bars
        bars = ax.bar(hourly_data['hour_of_day'], hourly_data['total_revenue'],
                     alpha=0.7, color='steelblue', label='Revenue')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12, color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot transactions as line
        line = ax2.plot(hourly_data['hour_of_day'], hourly_data['transaction_count'],
                       color='orange', marker='o', linewidth=2, label='Transactions')
        ax2.set_ylabel('Transaction Count', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title('Revenue and Transaction Volume by Hour', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 24))
        
        # Highlight peak hours if available
        if 'is_peak_hour' in hourly_data.columns:
            peak_hours = hourly_data[hourly_data['is_peak_hour'] == 1]['hour_of_day'].values
            for hour in peak_hours:
                ax.axvspan(hour-0.4, hour+0.4, alpha=0.2, color='red')
        
        # Add legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # 2. Average Transaction Value by Hour
        ax = axes[1, 0]
        
        bars = ax.bar(hourly_data['hour_of_day'], hourly_data['avg_transaction'],
                     color=sns.color_palette('coolwarm', len(hourly_data)))
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Transaction ($)', fontsize=12)
        ax.set_title('Average Transaction Value by Hour', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 24))
        
        # Add value labels on top bars
        for i, (idx, row) in enumerate(hourly_data.iterrows()):
            if row['avg_transaction'] == hourly_data['avg_transaction'].max():
                ax.text(row['hour_of_day'], row['avg_transaction'],
                       f"${row['avg_transaction']:.2f}", ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color='red')
        
        return self.save_figure(fig, 'time_hourly_demand.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_heatmap(self) -> Path:
        """
        Create hour-weekday heatmap visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating hour-weekday heatmap...")
        
        # Load data
        heatmap_data = self.load_csv_data('hour_weekday_heatmap.csv')
        
        # Check if data is empty
        if len(heatmap_data) == 0:
            self.logger.warning("hour_weekday_heatmap.csv is empty, creating simplified visualization...")
            # Create simplified chart from hourly detailed analysis instead
            hourly_data = self.load_csv_data('hourly_detailed_analysis.csv')
            
            if len(hourly_data) > 0:
                fig, ax = plt.subplots(figsize=(16, 8))
                
                # Create a simple bar chart instead of heatmap
                bars = ax.bar(hourly_data['hour_of_day'], hourly_data['transaction_count'],
                            color=sns.color_palette('YlOrRd', len(hourly_data)))
                ax.set_xlabel('Hour of Day', fontsize=12)
                ax.set_ylabel('Transaction Count', fontsize=12)
                ax.set_title('Hourly Transaction Pattern (Heatmap data not available)', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(range(0, 24))
                
                return self.save_figure(fig, 'time_demand_heatmap.png')
            else:
                self.logger.error("No hourly data available for fallback visualization")
                raise ValueError("No data available for heatmap visualization")
        
        # Set weekday as index
        if 'Weekday' in heatmap_data.columns:
            heatmap_data = heatmap_data.set_index('Weekday')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Transaction Count'},
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        ax.set_title('Transaction Demand Heatmap: Hour vs Weekday', fontsize=14, fontweight='bold')
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        return self.save_figure(fig, 'time_demand_heatmap.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_peak_analysis(self) -> Path:
        """
        Create peak vs off-peak analysis visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating peak analysis visualization...")
        
        # Load data
        peak_data = self.load_json_data('peak_offpeak_comparison.json')
        weekend_data = self.load_json_data('weekend_weekday_comparison.json')
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(16, 6))
        
        # 1. Peak vs Off-Peak Comparison
        ax = axes[0, 0]
        
        if peak_data:
            categories = list(peak_data.keys())
            revenue = [peak_data[cat]['total_revenue'] for cat in categories]
            transactions = [peak_data[cat]['transaction_count'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, revenue, width, label='Revenue', color='steelblue')
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, transactions, width, label='Transactions', color='orange')
            
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Revenue ($)', fontsize=12, color='steelblue')
            ax2.set_ylabel('Transaction Count', fontsize=12, color='orange')
            ax.set_title('Peak vs Off-Peak Period Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.tick_params(axis='y', labelcolor='steelblue')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. Weekend vs Weekday Comparison
        ax = axes[0, 1]
        
        if weekend_data:
            categories = list(weekend_data.keys())
            avg_daily_rev = [weekend_data[cat]['avg_daily_revenue'] for cat in categories]
            avg_daily_trans = [weekend_data[cat]['avg_daily_transactions'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, avg_daily_rev, width, label='Avg Daily Revenue', 
                          color='seagreen')
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, avg_daily_trans, width, label='Avg Daily Transactions',
                           color='coral')
            
            ax.set_xlabel('Day Type', fontsize=12)
            ax.set_ylabel('Average Daily Revenue ($)', fontsize=12, color='seagreen')
            ax2.set_ylabel('Average Daily Transactions', fontsize=12, color='coral')
            ax.set_title('Weekend vs Weekday Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.tick_params(axis='y', labelcolor='seagreen')
            ax2.tick_params(axis='y', labelcolor='coral')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return self.save_figure(fig, 'time_peak_analysis.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_all_visualizations(self) -> List[Path]:
        """
        Create all time-based demand visualizations.
        
        Returns:
            List of paths to saved figures
        """
        self.logger.info("=" * 80)
        self.logger.info("Creating Time-Based Demand Visualizations")
        self.logger.info("=" * 80)
        
        saved_files = []
        
        try:
            saved_files.append(self.plot_hourly_demand())
        except Exception as e:
            self.logger.error(f"Failed to create hourly demand plot: {e}")
        
        try:
            saved_files.append(self.plot_heatmap())
        except Exception as e:
            self.logger.error(f"Failed to create heatmap: {e}")
        
        try:
            saved_files.append(self.plot_peak_analysis())
        except Exception as e:
            self.logger.error(f"Failed to create peak analysis: {e}")
        
        self.logger.info(f"Created {len(saved_files)} time demand visualizations")
        
        return saved_files
