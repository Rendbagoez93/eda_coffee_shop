"""Sales Performance Visualizations (ARD Section 4.1)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_viz import BaseVisualizer
from ..utils.logger import log_execution_time, setup_logger


class SalesVisualizer(BaseVisualizer):
    """Create visualizations for sales performance analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize SalesVisualizer."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
    
    @log_execution_time(setup_logger(__name__))
    def plot_revenue_overview(self) -> Path:
        """
        Create revenue overview visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating revenue overview visualization...")
        
        # Load data
        metrics = self.load_json_data('sales_summary_metrics.json')
        daily_revenue = self.load_csv_data('daily_revenue.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(16, 12))
        
        # 1. Total Revenue KPI Card (top left)
        ax = axes[0, 0]
        ax.axis('off')
        
        total_rev = metrics['total_revenue']
        total_trans = metrics['total_transactions']
        avg_trans = metrics['average_transaction_value']
        
        kpi_text = f"""
        Total Revenue
        {self.format_currency(total_rev)}
        
        Total Transactions: {total_trans:,}
        Avg Transaction: {self.format_currency(avg_trans)}
        """
        
        ax.text(0.5, 0.5, kpi_text, ha='center', va='center',
                fontsize=18, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_title('Revenue Summary', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Daily Revenue Trend (top right)
        ax = axes[0, 1]
        daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
        
        ax.plot(daily_revenue['Date'], daily_revenue['total_revenue'], 
                linewidth=2, color='steelblue', label='Daily Revenue')
        
        if 'revenue_7day_ma' in daily_revenue.columns:
            ax.plot(daily_revenue['Date'], daily_revenue['revenue_7day_ma'],
                   linewidth=2, color='orange', linestyle='--', label='7-Day Moving Avg')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Top 10 Products by Revenue (bottom left)
        ax = axes[1, 0]
        products = self.load_csv_data('product_performance_detailed.csv')
        top_products = products.head(10).sort_values('total_revenue', ascending=True)
        
        bars = ax.barh(range(len(top_products)), top_products['total_revenue'], 
                       color=sns.color_palette('viridis', len(top_products)))
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products['coffee_name'], fontsize=10)
        ax.set_xlabel('Revenue ($)', fontsize=12)
        ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_products.iterrows()):
            ax.text(row['total_revenue'], i, f" ${row['total_revenue']:,.0f}",
                   va='center', fontsize=9)
        
        # 4. Revenue Distribution (bottom right)
        ax = axes[1, 1]
        
        if 'revenue_percentage' in products.columns:
            # Create pie chart for top products + others
            top_n = 8
            top_rev = products.head(top_n)
            others_rev = products.iloc[top_n:]['total_revenue'].sum()
            
            labels = list(top_rev['coffee_name']) + ['Others']
            sizes = list(top_rev['total_revenue']) + [others_rev]
            
            colors = sns.color_palette('Set3', len(labels))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax.set_title('Revenue Distribution by Product', fontsize=14, fontweight='bold')
        
        return self.save_figure(fig, 'sales_revenue_overview.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_weekday_analysis(self) -> Path:
        """
        Create weekday revenue analysis visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating weekday analysis visualization...")
        
        # Load daily revenue data
        daily_revenue = self.load_csv_data('daily_revenue.csv')
        daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(16, 6))
        
        # 1. Revenue by Weekday
        ax = axes[0, 0]
        weekday_avg = daily_revenue.groupby('day_name')['total_revenue'].mean()
        
        # Order by weekday
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_avg = weekday_avg.reindex([d for d in weekday_order if d in weekday_avg.index])
        
        bars = ax.bar(range(len(weekday_avg)), weekday_avg.values,
                      color=sns.color_palette('coolwarm', len(weekday_avg)))
        ax.set_xticks(range(len(weekday_avg)))
        ax.set_xticklabels(weekday_avg.index, rotation=45, ha='right')
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Average Revenue ($)', fontsize=12)
        ax.set_title('Average Revenue by Weekday', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(weekday_avg.values):
            ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Transaction Count by Weekday
        ax = axes[0, 1]
        weekday_count = daily_revenue.groupby('day_name')['transaction_count'].sum()
        weekday_count = weekday_count.reindex([d for d in weekday_order if d in weekday_count.index])
        
        bars = ax.bar(range(len(weekday_count)), weekday_count.values,
                      color=sns.color_palette('viridis', len(weekday_count)))
        ax.set_xticks(range(len(weekday_count)))
        ax.set_xticklabels(weekday_count.index, rotation=45, ha='right')
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Total Transactions', fontsize=12)
        ax.set_title('Total Transactions by Weekday', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(weekday_count.values):
            ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
        
        return self.save_figure(fig, 'sales_weekday_analysis.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_price_analysis(self) -> Path:
        """
        Create price analysis visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating price analysis visualization...")
        
        # Load data
        products = self.load_csv_data('product_performance_detailed.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(16, 6))
        
        # 1. Average Price by Product (top 15)
        ax = axes[0, 0]
        top_products = products.nlargest(15, 'avg_price').sort_values('avg_price', ascending=True)
        
        bars = ax.barh(range(len(top_products)), top_products['avg_price'],
                       color=sns.color_palette('mako', len(top_products)))
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products['coffee_name'], fontsize=9)
        ax.set_xlabel('Average Price ($)', fontsize=12)
        ax.set_title('Top 15 Products by Average Price', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_products.iterrows()):
            ax.text(row['avg_price'], i, f" ${row['avg_price']:.2f}",
                   va='center', fontsize=8)
        
        # 2. Price vs Sales Volume Scatter
        ax = axes[0, 1]
        
        scatter = ax.scatter(products['avg_price'], products['transaction_count'],
                           s=products['total_revenue']/100, alpha=0.6,
                           c=products['total_revenue'], cmap='viridis')
        
        ax.set_xlabel('Average Price ($)', fontsize=12)
        ax.set_ylabel('Sales Volume (Transactions)', fontsize=12)
        ax.set_title('Price vs Sales Volume\n(Bubble size = Total Revenue)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total Revenue ($)', fontsize=10)
        
        # Annotate top products
        top_5_by_revenue = products.nlargest(5, 'total_revenue')
        for idx, row in top_5_by_revenue.iterrows():
            ax.annotate(row['coffee_name'], 
                       xy=(row['avg_price'], row['transaction_count']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        return self.save_figure(fig, 'sales_price_analysis.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_all_visualizations(self) -> List[Path]:
        """
        Create all sales performance visualizations.
        
        Returns:
            List of paths to saved figures
        """
        self.logger.info("=" * 80)
        self.logger.info("Creating Sales Performance Visualizations")
        self.logger.info("=" * 80)
        
        saved_files = []
        
        try:
            saved_files.append(self.plot_revenue_overview())
        except Exception as e:
            self.logger.error(f"Failed to create revenue overview: {e}")
        
        try:
            saved_files.append(self.plot_weekday_analysis())
        except Exception as e:
            self.logger.error(f"Failed to create weekday analysis: {e}")
        
        try:
            saved_files.append(self.plot_price_analysis())
        except Exception as e:
            self.logger.error(f"Failed to create price analysis: {e}")
        
        self.logger.info(f"Created {len(saved_files)} sales visualizations")
        
        return saved_files
