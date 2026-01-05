"""Payment Behavior Visualizations (ARD Section 4.4)."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_viz import BaseVisualizer
from ..utils.logger import log_execution_time, setup_logger


class PaymentVisualizer(BaseVisualizer):
    """Create visualizations for payment behavior analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize PaymentVisualizer."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
    
    @log_execution_time(setup_logger(__name__))
    def plot_payment_distribution(self) -> Path:
        """
        Create payment method distribution visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating payment distribution visualization...")
        
        # Load data
        payment_dist = self.load_csv_data('payment_distribution.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(16, 6))
        
        # 1. Payment Method Distribution (Pie Chart)
        ax = axes[0, 0]
        
        colors = sns.color_palette('pastel', len(payment_dist))
        
        wedges, texts, autotexts = ax.pie(payment_dist['transaction_count'],
                                          labels=payment_dist['cash_type'],
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=[0.05] * len(payment_dist))
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title('Payment Method Distribution by Transaction Count',
                    fontsize=14, fontweight='bold')
        
        # 2. Payment Method Metrics Comparison
        ax = axes[0, 1]
        
        x = np.arange(len(payment_dist))
        width = 0.25
        
        bars1 = ax.bar(x - width, payment_dist['transaction_count'], width,
                      label='Transactions', color='steelblue')
        bars2 = ax.bar(x, payment_dist['total_revenue']/100, width,
                      label='Revenue (÷100)', color='lightgreen')
        bars3 = ax.bar(x + width, payment_dist['avg_transaction'], width,
                      label='Avg Transaction', color='coral')
        
        ax.set_xlabel('Payment Method', fontsize=12)
        ax.set_ylabel('Value (scaled)', fontsize=12)
        ax.set_title('Payment Method Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(payment_dist['cash_type'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        return self.save_figure(fig, 'payment_distribution.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_spending_patterns(self) -> Path:
        """
        Create spending pattern analysis visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating spending patterns visualization...")
        
        # Load data
        payment_dist = self.load_csv_data('payment_distribution.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 2, figsize=(16, 12))
        
        # 1. Revenue Distribution by Payment Method
        ax = axes[0, 0]
        
        if len(payment_dist) > 0:
            colors = sns.color_palette('Set2', len(payment_dist))
            bars = ax.barh(payment_dist['cash_type'], payment_dist['total_revenue'], color=colors)
            
            ax.set_xlabel('Total Revenue ($)', fontsize=12)
            ax.set_ylabel('Payment Method', fontsize=12)
            ax.set_title('Revenue by Payment Method', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels and percentages
            total_revenue = payment_dist['total_revenue'].sum()
            for i, (idx, row) in enumerate(payment_dist.iterrows()):
                width = bars[i].get_width()
                pct = (row['total_revenue'] / total_revenue) * 100
                ax.text(width, bars[i].get_y() + bars[i].get_height()/2.,
                       f' ${width:,.0f} ({pct:.1f}%)',
                       ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 2. Average Transaction Value by Payment Method
        ax = axes[0, 1]
        
        if len(payment_dist) > 0:
            colors = sns.color_palette('coolwarm', len(payment_dist))
            bars = ax.bar(payment_dist['cash_type'], payment_dist['avg_transaction'], color=colors)
            
            ax.set_xlabel('Payment Method', fontsize=12)
            ax.set_ylabel('Average Transaction ($)', fontsize=12)
            ax.set_title('Average Spending by Payment Method', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.2f}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
            
            # Highlight highest
            max_idx = payment_dist['avg_transaction'].idxmax()
            max_pos = list(payment_dist.index).index(max_idx)
            bars[max_pos].set_edgecolor('red')
            bars[max_pos].set_linewidth(3)
        
        # 3. Cash vs Cashless Comparison
        ax = axes[1, 0]
        
        cash_comparison = self.load_json_data('cash_cashless_comparison.json')
        
        if cash_comparison:
            categories = list(cash_comparison.keys())
            
            metrics = ['transaction_count', 'total_revenue', 'avg_transaction']
            metric_labels = ['Transactions', 'Revenue (÷100)', 'Avg Transaction']
            
            x = np.arange(len(categories))
            width = 0.25
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [cash_comparison[cat][metric] for cat in categories]
                
                # Scale revenue for better visualization
                if metric == 'total_revenue':
                    values = [v / 100 for v in values]
                
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, label=label)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    if metric == 'total_revenue':
                        label_text = f'${val*100:,.0f}'
                    elif metric == 'avg_transaction':
                        label_text = f'${val:.2f}'
                    else:
                        label_text = f'{val:,.0f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label_text, ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Payment Category', fontsize=12)
            ax.set_ylabel('Value (scaled)', fontsize=12)
            ax.set_title('Cash vs Cashless Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Transaction Count Distribution
        ax = axes[1, 1]
        
        if len(payment_dist) > 0:
            # Create donut chart
            colors = sns.color_palette('husl', len(payment_dist))
            
            wedges, texts, autotexts = ax.pie(payment_dist['transaction_count'], 
                                              labels=payment_dist['cash_type'],
                                              colors=colors, autopct='%1.1f%%',
                                              startangle=90, pctdistance=0.85)
            
            # Draw center circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            for text in texts:
                text.set_fontsize(11)
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            ax.set_title('Transaction Volume Distribution', fontsize=14, fontweight='bold')
        
        return self.save_figure(fig, 'payment_spending_patterns.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_payment_trends(self) -> Path:
        """
        Create payment trends visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating payment trends visualization...")
        
        # Load data
        payment_hourly = self.load_csv_data('payment_hourly_patterns.csv')
        payment_weekday = self.load_csv_data('payment_weekday_patterns.csv')
        
        # Create figure
        fig, axes = self.create_grid_layout(2, 1, figsize=(16, 12))
        
        # 1. Payment Method Usage by Hour
        ax = axes[0, 0]
        
        # Data is already in the right format (hour_of_day as index, payment types as columns)
        if len(payment_hourly) > 0:
            if 'hour_of_day' in payment_hourly.columns:
                hourly_pivot = payment_hourly.set_index('hour_of_day')
            else:
                hourly_pivot = payment_hourly
            
            # Check if we have numeric columns to plot
            numeric_cols = hourly_pivot.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                hourly_pivot[numeric_cols].plot(kind='line', ax=ax, marker='o', linewidth=2)
                
                ax.set_xlabel('Hour of Day', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
                ax.set_title('Payment Method Preference by Hour', fontsize=14, fontweight='bold')
                ax.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            else:
                # No numeric data, create placeholder
                ax.text(0.5, 0.5, 'Payment hourly data not available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Payment Method Preference by Hour', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No hourly payment data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Payment Method Preference by Hour', fontsize=14, fontweight='bold')
        
        # 2. Payment Method Usage by Weekday (Stacked Bar)
        ax = axes[1, 0]
        
        if len(payment_weekday) > 0:
            # Pivot data
            weekday_pivot = payment_weekday.pivot(index='Weekday',
                                                 columns='cash_type',
                                                 values='transaction_count')
            
            # Define weekday order - check if data uses abbreviations or full names
            if any(day in weekday_pivot.index for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
                weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            else:
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            weekday_pivot = weekday_pivot.reindex([day for day in weekday_order if day in weekday_pivot.index])
            
            # Plot to the specified axes
            weekday_pivot.plot(kind='bar', stacked=True, ax=ax)
            
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Transaction Count', fontsize=12)
            ax.set_title('Payment Method Distribution by Weekday', fontsize=14, fontweight='bold')
            ax.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No weekday payment data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Payment Method Distribution by Weekday', fontsize=14, fontweight='bold')
        
        return self.save_figure(fig, 'payment_trends.png')
    
    @log_execution_time(setup_logger(__name__))
    def plot_payment_product_relationship(self) -> Path:
        """
        Create payment method and product relationship visualization.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating payment-product relationship visualization...")
        
        # Load data - check if file exists
        try:
            payment_product = self.load_csv_data('payment_product_analysis.csv')
        except FileNotFoundError:
            self.logger.warning("payment_product_analysis.csv not found, creating placeholder visualization")
            # Create a simple placeholder figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'Payment-Product Analysis\nData not available\n\nThis visualization requires payment_product_analysis.csv',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Payment-Product Relationship', fontsize=14, fontweight='bold')
            ax.axis('off')
            return self.save_figure(fig, 'payment_product_relationship.png')
        
        # Create figure
        fig, axes = self.create_grid_layout(1, 2, figsize=(18, 6))
        
        # 1. Heatmap: Product vs Payment Method
        ax = axes[0, 0]
        
        # Pivot data
        heatmap_data = payment_product.pivot(index='coffee_name',
                                             columns='cash_type',
                                             values='transaction_count')
        
        # Sort by total transactions
        heatmap_data['total'] = heatmap_data.sum(axis=1)
        heatmap_data = heatmap_data.sort_values('total', ascending=False).drop('total', axis=1)
        
        # Take top products
        top_products = heatmap_data.head(12)
        
        sns.heatmap(top_products, annot=True, fmt='.0f', cmap='YlGnBu',
                   cbar_kws={'label': 'Transaction Count'},
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('Payment Method', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Product-Payment Method Matrix (Top 12 Products)',
                    fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # 2. Payment Preference by Product Category
        ax = axes[0, 1]
        
        # Calculate payment preference percentage for each product
        payment_pct = payment_product.pivot(index='coffee_name',
                                           columns='cash_type',
                                           values='transaction_count')
        
        # Normalize to percentages
        payment_pct = payment_pct.div(payment_pct.sum(axis=1), axis=0) * 100
        
        # Get top products by total transactions
        top_product_names = heatmap_data.head(10).index
        payment_pct_top = payment_pct.loc[top_product_names]
        
        payment_pct_top.plot(kind='barh', stacked=True, ax=ax)
        
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_ylabel('Coffee Product', fontsize=12)
        ax.set_title('Payment Method Distribution by Product (%)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')
        
        return self.save_figure(fig, 'payment_product_relationship.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_all_visualizations(self) -> List[Path]:
        """
        Create all payment behavior visualizations.
        
        Returns:
            List of paths to saved figures
        """
        self.logger.info("=" * 80)
        self.logger.info("Creating Payment Behavior Visualizations")
        self.logger.info("=" * 80)
        
        saved_files = []
        
        try:
            saved_files.append(self.plot_payment_distribution())
        except Exception as e:
            self.logger.error(f"Failed to create payment distribution plot: {e}")
        
        try:
            saved_files.append(self.plot_spending_patterns())
        except Exception as e:
            self.logger.error(f"Failed to create spending patterns plot: {e}")
        
        try:
            saved_files.append(self.plot_payment_trends())
        except Exception as e:
            self.logger.error(f"Failed to create payment trends plot: {e}")
        
        try:
            saved_files.append(self.plot_payment_product_relationship())
        except Exception as e:
            self.logger.error(f"Failed to create payment-product relationship plot: {e}")
        
        self.logger.info(f"Created {len(saved_files)} payment behavior visualizations")
        
        return saved_files
