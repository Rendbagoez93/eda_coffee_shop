"""Seasonality and Trends Visualizations (ARD Section 4.5)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_viz import BaseVisualizer
from ..utils.logger import log_execution_time, setup_logger


class SeasonalityVisualizer(BaseVisualizer):
    """Create visualizations for seasonality and trend analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize SeasonalityVisualizer."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
    
    @log_execution_time(setup_logger(__name__))
    def plot_monthly_trends(self) -> Path:
        """
        Create monthly trend visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating monthly trends visualization...")
        
        # Load data
        monthly_data = self.load_csv_data('monthly_trends_analysis.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(18, 12))
        
        # 1. Revenue and Transactions by Month
        ax = axes[0, 0]
        
        # Define month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        # Reindex if Month_name exists
        if 'Month_name' in monthly_data.columns:
            monthly_data = monthly_data.set_index('Month_name')
            monthly_data = monthly_data.reindex([m for m in month_order if m in monthly_data.index])
            monthly_data = monthly_data.reset_index()
        
        x = range(len(monthly_data))
        
        ax.bar(x, monthly_data['total_revenue'], alpha=0.7, color='steelblue',
              label='Revenue')
        
        ax2 = ax.twinx()
        ax2.plot(x, monthly_data['transaction_count'], color='orange', marker='o',
                linewidth=2, markersize=8, label='Transactions')
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12, color='steelblue')
        ax2.set_ylabel('Transaction Count', fontsize=12, color='orange')
        ax.set_title('Monthly Revenue and Transaction Trends', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                          rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. Average Transaction Value by Month
        ax = axes[0, 1]
        
        colors = sns.color_palette('coolwarm', len(monthly_data))
        bars = ax.bar(x, monthly_data['avg_transaction'], color=colors)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Transaction ($)', fontsize=12)
        ax.set_title('Average Transaction Value by Month', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight highest and lowest
        max_idx = monthly_data['avg_transaction'].idxmax()
        min_idx = monthly_data['avg_transaction'].idxmin()
        
        if max_idx in monthly_data.index:
            max_pos = monthly_data.index.get_loc(max_idx)
            bars[max_pos].set_edgecolor('green')
            bars[max_pos].set_linewidth(3)
        
        if min_idx in monthly_data.index:
            min_pos = monthly_data.index.get_loc(min_idx)
            bars[min_pos].set_edgecolor('red')
            bars[min_pos].set_linewidth(3)
        
        # 3. Monthly Growth Rate
        ax = axes[1, 0]
        
        if 'revenue_growth_rate' in monthly_data.columns:
            growth_colors = ['green' if g > 0 else 'red' 
                           for g in monthly_data['revenue_growth_rate'].fillna(0)]
            
            bars = ax.bar(x, monthly_data['revenue_growth_rate'], color=growth_colors,
                         alpha=0.7)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Growth Rate (%)', fontsize=12)
            ax.set_title('Month-over-Month Revenue Growth Rate', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (idx, row) in enumerate(monthly_data.iterrows()):
                if pd.notna(row.get('revenue_growth_rate')):
                    value = row['revenue_growth_rate']
                    ax.text(i, value, f'{value:.1f}%',
                           ha='center', va='bottom' if value > 0 else 'top',
                           fontsize=8, fontweight='bold')
        
        # 4. Cumulative Revenue
        ax = axes[1, 1]
        
        cumulative_revenue = monthly_data['total_revenue'].cumsum()
        
        ax.plot(x, cumulative_revenue, marker='o', linewidth=2.5,
               color='darkgreen', markersize=8)
        ax.fill_between(x, cumulative_revenue, alpha=0.3, color='lightgreen')
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Cumulative Revenue ($)', fontsize=12)
        ax.set_title('Cumulative Revenue Growth', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add final value
        final_revenue = cumulative_revenue.iloc[-1]
        ax.text(x[-1], final_revenue, f'${final_revenue:,.0f}',
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return self.save_figure(fig, 'seasonality_monthly_trends.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_seasonal_patterns(self) -> Path:
        """
        Create seasonal pattern visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating seasonal patterns visualization...")
        
        # Load data
        seasonal_data = self.load_json_data('seasonal_patterns.json')
        monthly_data = self.load_csv_data('monthly_trends_analysis.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(18, 12))
        
        # 1. Seasonal Revenue Comparison
        ax = axes[0, 0]
        
        if seasonal_data and 'by_season' in seasonal_data:
            seasons = list(seasonal_data['by_season'].keys())
            revenues = [seasonal_data['by_season'][s]['total_revenue'] for s in seasons]
            transactions = [seasonal_data['by_season'][s]['transaction_count'] for s in seasons]
            
            x = np.arange(len(seasons))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, revenues, width, label='Revenue',
                          color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, transactions, width, label='Transactions',
                           color=['steelblue', 'seagreen', 'darkorange', 'darkred'])
            
            ax.set_xlabel('Season', fontsize=12)
            ax.set_ylabel('Revenue ($)', fontsize=12, color='black')
            ax2.set_ylabel('Transaction Count', fontsize=12, color='black')
            ax.set_title('Seasonal Revenue and Transaction Comparison',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(seasons)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. Seasonal Market Share
        ax = axes[0, 1]
        
        if seasonal_data and 'by_season' in seasonal_data:
            seasons = list(seasonal_data['by_season'].keys())
            revenues = [seasonal_data['by_season'][s]['total_revenue'] for s in seasons]
            
            colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
            
            wedges, texts, autotexts = ax.pie(revenues, labels=seasons, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax.set_title('Revenue Distribution by Season', fontsize=14, fontweight='bold')
        
        # 3. Day of Week Pattern
        ax = axes[1, 0]
        
        if seasonal_data and 'by_weekday' in seasonal_data:
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_data = seasonal_data['by_weekday']
            
            # Reorder based on weekdays
            ordered_revenues = [weekday_data.get(day, {}).get('avg_daily_revenue', 0) 
                              for day in weekdays if day in weekday_data]
            ordered_days = [day for day in weekdays if day in weekday_data]
            
            colors = ['steelblue' if day not in ['Saturday', 'Sunday'] else 'coral'
                     for day in ordered_days]
            
            bars = ax.bar(range(len(ordered_days)), ordered_revenues, color=colors)
            
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Average Daily Revenue ($)', fontsize=12)
            ax.set_title('Average Revenue by Day of Week', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(ordered_days)))
            ax.set_xticklabels(ordered_days, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='steelblue', label='Weekday'),
                             Patch(facecolor='coral', label='Weekend')]
            ax.legend(handles=legend_elements, loc='upper right')
        
        # 4. Monthly Variance Analysis
        ax = axes[1, 1]
        
        if len(monthly_data) > 0:
            # Calculate statistics
            month_revenue = monthly_data['total_revenue']
            
            boxplot = ax.boxplot([month_revenue], vert=True, patch_artist=True,
                                labels=['Monthly Revenue'])
            
            for patch in boxplot['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            # Add scatter of actual values
            y = month_revenue
            x = np.random.normal(1, 0.04, len(y))
            ax.scatter(x, y, alpha=0.5, color='darkblue', s=100)
            
            ax.set_ylabel('Revenue ($)', fontsize=12)
            ax.set_title('Monthly Revenue Variance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            stats_text = f"Mean: ${month_revenue.mean():,.0f}\n"
            stats_text += f"Median: ${month_revenue.median():,.0f}\n"
            stats_text += f"Std Dev: ${month_revenue.std():,.0f}\n"
            stats_text += f"CV: {(month_revenue.std()/month_revenue.mean()*100):.1f}%"
            
            ax.text(1.15, month_revenue.max(), stats_text,
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return self.save_figure(fig, 'seasonality_patterns.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_growth_analysis(self) -> Path:
        """
        Create growth trend analysis visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating growth analysis visualization...")
        
        # Load data
        growth_data = self.load_json_data('growth_trends.json')
        monthly_data = self.load_csv_data('monthly_trends_analysis.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(18, 12))
        
        # 1. Overall Growth Metrics
        ax = axes[0, 0]
        
        if growth_data and 'overall_growth' in growth_data:
            metrics = growth_data['overall_growth']
            
            metric_names = []
            metric_values = []
            
            if 'total_revenue_growth' in metrics:
                metric_names.append('Revenue\nGrowth')
                metric_values.append(metrics['total_revenue_growth'])
            
            if 'transaction_growth' in metrics:
                metric_names.append('Transaction\nGrowth')
                metric_values.append(metrics['transaction_growth'])
            
            if 'avg_transaction_growth' in metrics:
                metric_names.append('Avg Transaction\nGrowth')
                metric_values.append(metrics['avg_transaction_growth'])
            
            if metric_names:
                colors = ['green' if v > 0 else 'red' for v in metric_values]
                bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.set_ylabel('Growth Rate (%)', fontsize=12)
                ax.set_title('Overall Growth Metrics (First vs Last Month)',
                            fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center',
                           va='bottom' if value > 0 else 'top',
                           fontsize=11, fontweight='bold')
        
        # 2. Revenue Trend with Trendline
        ax = axes[0, 1]
        
        if len(monthly_data) > 0:
            x = range(len(monthly_data))
            y = monthly_data['total_revenue'].values
            
            # Plot actual values
            ax.plot(x, y, marker='o', linewidth=2, markersize=8,
                   color='steelblue', label='Actual Revenue')
            
            # Add trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=2, label='Trend Line', alpha=0.7)
            
            # Shade confidence interval
            ax.fill_between(x, y * 0.95, y * 1.05, alpha=0.2, color='blue')
            
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Revenue ($)', fontsize=12)
            ax.set_title('Revenue Trend Analysis', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add trend direction annotation
            slope = z[0]
            trend_text = "Upward Trend" if slope > 0 else "Downward Trend"
            trend_color = "green" if slope > 0 else "red"
            
            ax.text(0.05, 0.95, trend_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color=trend_color,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Transaction Volume Trend
        ax = axes[1, 0]
        
        if len(monthly_data) > 0:
            x = range(len(monthly_data))
            y = monthly_data['transaction_count'].values
            
            ax.bar(x, y, alpha=0.7, color='orange', label='Transaction Count')
            
            # Add trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=2, label='Trend Line')
            
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Transaction Count', fontsize=12)
            ax.set_title('Transaction Volume Trend', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. YoY Comparison (if available)
        ax = axes[1, 1]
        
        if growth_data and 'monthly_comparison' in growth_data:
            comparison = growth_data['monthly_comparison']
            
            months = list(comparison.keys())[:6]  # Show first 6 months
            growth_rates = [comparison[m].get('revenue_growth', 0) for m in months]
            
            colors = ['green' if g > 0 else 'red' for g in growth_rates]
            bars = ax.bar(range(len(months)), growth_rates, color=colors, alpha=0.7)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Growth Rate (%)', fontsize=12)
            ax.set_title('Month-over-Month Growth Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(months)))
            ax.set_xticklabels(months, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, growth_rates)):
                height = bar.get_height()
                ax.text(i, height, f'{value:.1f}%',
                       ha='center', va='bottom' if value > 0 else 'top',
                       fontsize=9, fontweight='bold')
        else:
            # If no comparison data, show alternative metric
            if len(monthly_data) > 1:
                monthly_data['pct_change'] = monthly_data['total_revenue'].pct_change() * 100
                
                x = range(1, len(monthly_data))
                y = monthly_data['pct_change'].iloc[1:].values
                
                colors = ['green' if g > 0 else 'red' for g in y]
                bars = ax.bar(x, y, color=colors, alpha=0.7)
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('% Change', fontsize=12)
                ax.set_title('Month-over-Month % Change', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index).iloc[1:],
                                  rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
        
        return self.save_figure(fig, 'seasonality_growth.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_all_visualizations(self) -> List[Path]:
        """
        Create all seasonality and trend visualizations.
        
        Returns:
            List of paths to saved figures
        """
        self.logger.info("=" * 80)
        self.logger.info("Creating Seasonality and Trend Visualizations")
        self.logger.info("=" * 80)
        
        saved_files = []
        
        try:
            saved_files.append(self.plot_monthly_trends())
        except Exception as e:
            self.logger.error(f"Failed to create monthly trends plot: {e}")
        
        try:
            saved_files.append(self.plot_seasonal_patterns())
        except Exception as e:
            self.logger.error(f"Failed to create seasonal patterns plot: {e}")
        
        try:
            saved_files.append(self.plot_growth_analysis())
        except Exception as e:
            self.logger.error(f"Failed to create growth analysis plot: {e}")
        
        self.logger.info(f"Created {len(saved_files)} seasonality visualizations")
        
        return saved_files
