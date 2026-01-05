"""Product Preference Visualizations (ARD Section 4.3)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_viz import BaseVisualizer
from ..utils.logger import log_execution_time, setup_logger


class ProductVisualizer(BaseVisualizer):
    """Create visualizations for product preference analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize ProductVisualizer."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
    
    @log_execution_time(setup_logger(__name__))
    def plot_product_popularity(self) -> Path:
        """
        Create product popularity visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating product popularity visualization...")
        
        # Load data
        coffee_counts = self.load_csv_data('coffee_counts.csv')
        product_perf = self.load_csv_data('product_performance_detailed.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(18, 12))
        
        # 1. Transaction Count by Product
        ax = axes[0, 0]
        
        sorted_counts = coffee_counts.sort_values('sales_count', ascending=False)
        colors = sns.color_palette('viridis', len(sorted_counts))
        
        bars = ax.barh(sorted_counts['coffee_name'], sorted_counts['sales_count'], color=colors)
        ax.set_xlabel('Transaction Count', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Product Popularity by Transaction Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width):,}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Revenue by Product
        ax = axes[0, 1]
        
        sorted_revenue = product_perf.sort_values('total_revenue', ascending=False)
        colors = sns.color_palette('plasma', len(sorted_revenue))
        
        bars = ax.barh(sorted_revenue['coffee_name'], sorted_revenue['total_revenue'], color=colors)
        ax.set_xlabel('Total Revenue ($)', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Product Revenue Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'${width:,.0f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 3. Average Price by Product
        ax = axes[1, 0]
        
        sorted_price = product_perf.sort_values('avg_price', ascending=False)
        colors = sns.color_palette('coolwarm', len(sorted_price))
        
        bars = ax.barh(sorted_price['coffee_name'], sorted_price['avg_price'], color=colors)
        ax.set_xlabel('Average Price ($)', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Product Pricing Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'${width:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 4. Market Share Pie Chart
        ax = axes[1, 1]
        
        # Calculate market share based on transaction count
        sorted_counts = coffee_counts.sort_values('sales_count', ascending=False)
        
        # Show top products and group others
        top_n = 8
        if len(sorted_counts) > top_n:
            top_products = sorted_counts.head(top_n)
            others_count = sorted_counts.tail(len(sorted_counts) - top_n)['sales_count'].sum()
            
            labels = list(top_products['coffee_name']) + ['Others']
            sizes = list(top_products['sales_count']) + [others_count]
        else:
            labels = sorted_counts['coffee_name']
            sizes = sorted_counts['sales_count']
        
        colors = sns.color_palette('Set3', len(labels))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        
        # Improve text readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title('Market Share by Transaction Count', fontsize=14, fontweight='bold')
        
        return self.save_figure(fig, 'product_popularity.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_time_of_day_patterns(self) -> Path:
        """
        Create time-of-day product preference visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating time-of-day patterns visualization...")
        
        # Load data
        time_of_day_data = self.load_csv_data('product_time_of_day_analysis.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(18, 6))
        
        # 1. Product Sales by Time of Day (Grouped Bar)
        ax = axes[0, 0]
        
        # Pivot data for grouped bar chart
        pivot_data = time_of_day_data.pivot(index='coffee_name', 
                                            columns='Time_of_Day', 
                                            values='sales_count')
        
        # Sort by total transactions
        pivot_data['total'] = pivot_data.sum(axis=1)
        pivot_data = pivot_data.sort_values('total', ascending=False).drop('total', axis=1)
        
        # Take top products
        top_products = pivot_data.head(10)
        
        top_products.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Coffee Product', fontsize=12)
        ax.set_ylabel('Transaction Count', fontsize=12)
        ax.set_title('Product Popularity by Time of Day', fontsize=14, fontweight='bold')
        ax.legend(title='Time of Day', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Product Revenue by Time of Day (Stacked Area)
        ax = axes[0, 1]
        
        # Pivot revenue data
        revenue_pivot = time_of_day_data.pivot(index='Time_of_Day',
                                              columns='coffee_name',
                                              values='total_revenue')
        
        # Get top products by total revenue
        product_revenue = revenue_pivot.sum().sort_values(ascending=False)
        top_revenue_products = product_revenue.head(8).index
        
        # Create stacked area plot
        revenue_pivot[top_revenue_products].plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        
        ax.set_xlabel('Time of Day', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue Composition by Time of Day', fontsize=14, fontweight='bold')
        ax.legend(title='Coffee Product', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        return self.save_figure(fig, 'product_time_patterns.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_hourly_patterns(self) -> Path:
        """
        Create hourly product preference heatmap.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating hourly patterns visualization...")
        
        # Load data
        hourly_data = self.load_csv_data('product_hourly_patterns.csv')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Data is already in the right format (coffee_name as index, hours as columns)
        if 'coffee_name' in hourly_data.columns:
            heatmap_data = hourly_data.set_index('coffee_name')
        else:
            heatmap_data = hourly_data
        
        # Fill NaN with 0
        heatmap_data = heatmap_data.fillna(0)
        
        # Sort by total transactions
        heatmap_data['total'] = heatmap_data.sum(axis=1)
        heatmap_data = heatmap_data.sort_values('total', ascending=False).drop('total', axis=1)
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False,
                   cbar_kws={'label': 'Transaction Count'},
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Product Demand Patterns by Hour', fontsize=14, fontweight='bold')
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        return self.save_figure(fig, 'product_hourly_heatmap.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_product_metrics(self) -> Path:
        """
        Create comprehensive product metrics visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating product metrics visualization...")
        
        # Load data
        product_perf = self.load_csv_data('product_performance_detailed.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(18, 12))
        
        # 1. Scatter: Price vs Popularity
        ax = axes[0, 0]
        
        scatter = ax.scatter(product_perf['avg_price'], 
                           product_perf['transaction_count'],
                           s=product_perf['total_revenue']/10,
                           c=product_perf['total_revenue'],
                           cmap='viridis', alpha=0.6, edgecolors='black')
        
        # Add product labels
        for idx, row in product_perf.iterrows():
            ax.annotate(row['coffee_name'], 
                       (row['avg_price'], row['transaction_count']),
                       fontsize=8, alpha=0.7, ha='center')
        
        ax.set_xlabel('Average Price ($)', fontsize=12)
        ax.set_ylabel('Transaction Count', fontsize=12)
        ax.set_title('Price vs Popularity (bubble size = revenue)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total Revenue ($)', fontsize=10)
        
        # 2. Revenue Concentration
        ax = axes[0, 1]
        
        sorted_revenue = product_perf.sort_values('total_revenue', ascending=False)
        sorted_revenue['cumulative_pct'] = (sorted_revenue['total_revenue'].cumsum() / 
                                            sorted_revenue['total_revenue'].sum() * 100)
        
        x = range(len(sorted_revenue))
        
        ax.bar(x, sorted_revenue['total_revenue'], color='steelblue', alpha=0.7,
              label='Revenue')
        
        ax2 = ax.twinx()
        ax2.plot(x, sorted_revenue['cumulative_pct'], color='red', marker='o',
                linewidth=2, label='Cumulative %')
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% Line')
        
        ax.set_xlabel('Product Rank', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12, color='steelblue')
        ax2.set_ylabel('Cumulative %', fontsize=12, color='red')
        ax.set_title('Revenue Concentration (Pareto Analysis)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_revenue['coffee_name'], rotation=45, ha='right', fontsize=9)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. Transaction Frequency Distribution
        ax = axes[1, 0]
        
        sorted_trans = product_perf.sort_values('transaction_count', ascending=False)
        
        colors = ['green' if tc > sorted_trans['transaction_count'].median() else 'lightcoral'
                 for tc in sorted_trans['transaction_count']]
        
        bars = ax.bar(range(len(sorted_trans)), sorted_trans['transaction_count'], color=colors)
        ax.set_xlabel('Product', fontsize=12)
        ax.set_ylabel('Transaction Count', fontsize=12)
        ax.set_title('Transaction Frequency Distribution', fontsize=14, fontweight='bold')
        ax.axhline(y=sorted_trans['transaction_count'].median(), color='black',
                  linestyle='--', label=f"Median: {sorted_trans['transaction_count'].median():.0f}")
        ax.set_xticks(range(len(sorted_trans)))
        ax.set_xticklabels(sorted_trans['coffee_name'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Price Tiers Analysis
        ax = axes[1, 1]
        
        # Create price tiers
        product_perf['price_tier'] = pd.cut(product_perf['avg_price'],
                                           bins=[0, 3, 4, 5, 100],
                                           labels=['Budget', 'Standard', 'Premium', 'Luxury'])
        
        tier_analysis = product_perf.groupby('price_tier', observed=True).agg({
            'transaction_count': 'sum',
            'total_revenue': 'sum',
            'coffee_name': 'count'
        }).reset_index()
        tier_analysis.columns = ['price_tier', 'total_transactions', 'total_revenue', 'product_count']
        
        x = np.arange(len(tier_analysis))
        width = 0.25
        
        bars1 = ax.bar(x - width, tier_analysis['product_count'], width, label='Product Count',
                      color='lightblue')
        bars2 = ax.bar(x, tier_analysis['total_transactions']/100, width, 
                      label='Transactions (÷100)', color='lightgreen')
        bars3 = ax.bar(x + width, tier_analysis['total_revenue']/100, width,
                      label='Revenue (÷100)', color='salmon')
        
        ax.set_xlabel('Price Tier', fontsize=12)
        ax.set_ylabel('Count / Value (scaled)', fontsize=12)
        ax.set_title('Performance by Price Tier', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tier_analysis['price_tier'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return self.save_figure(fig, 'product_metrics.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_all_visualizations(self) -> List[Path]:
        """
        Create all product preference visualizations.
        
        Returns:
            List of paths to saved figures
        """
        self.logger.info("=" * 80)
        self.logger.info("Creating Product Preference Visualizations")
        self.logger.info("=" * 80)
        
        saved_files = []
        
        try:
            saved_files.append(self.plot_product_popularity())
        except Exception as e:
            self.logger.error(f"Failed to create product popularity plot: {e}")
        
        try:
            saved_files.append(self.plot_time_of_day_patterns())
        except Exception as e:
            self.logger.error(f"Failed to create time-of-day patterns plot: {e}")
        
        try:
            saved_files.append(self.plot_hourly_patterns())
        except Exception as e:
            self.logger.error(f"Failed to create hourly patterns plot: {e}")
        
        try:
            saved_files.append(self.plot_product_metrics())
        except Exception as e:
            self.logger.error(f"Failed to create product metrics plot: {e}")
        
        self.logger.info(f"Created {len(saved_files)} product preference visualizations")
        
        return saved_files
