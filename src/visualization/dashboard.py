"""Comprehensive Dashboard Generator for Coffee Sales Analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .base_viz import BaseVisualizer
from .sales_viz import SalesVisualizer
from .time_viz import TimeVisualizer
from .product_viz import ProductVisualizer
from .payment_viz import PaymentVisualizer
from .seasonality_viz import SeasonalityVisualizer
from ..utils.logger import log_execution_time, setup_logger


class DashboardGenerator(BaseVisualizer):
    """Generate comprehensive dashboards combining all analyses."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize DashboardGenerator."""
        super().__init__(config_dir)
        self.logger = setup_logger(__name__)
        
        # Initialize all visualizers
        self.sales_viz = SalesVisualizer(config_dir)
        self.time_viz = TimeVisualizer(config_dir)
        self.product_viz = ProductVisualizer(config_dir)
        self.payment_viz = PaymentVisualizer(config_dir)
        self.seasonality_viz = SeasonalityVisualizer(config_dir)
    
    @log_execution_time(setup_logger(__name__))
    def create_executive_summary(self) -> Path:
        """
        Create executive summary dashboard with key metrics.
        
        Returns:
            Path to saved figure
        """
        self.logger.info("Creating executive summary dashboard...")
        
        # Load key data
        sales_summary = self.load_json_data('sales_summary_metrics.json')
        daily_revenue = self.load_csv_data('daily_revenue.csv')
        coffee_counts = self.load_csv_data('coffee_counts.csv')
        payment_dist = self.load_csv_data('payment_distribution.csv')
        monthly_data = self.load_csv_data('monthly_trends_analysis.csv')
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Coffee Shop Sales Analysis - Executive Summary',
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Key Metrics Cards (Top Row)
        metrics_data = []
        
        if sales_summary:
            if 'total_revenue' in sales_summary:
                metrics_data.append(('Total Revenue', f"${sales_summary['total_revenue']:,.0f}"))
            if 'total_transactions' in sales_summary:
                metrics_data.append(('Total Transactions', f"{sales_summary['total_transactions']:,}"))
            if 'avg_transaction_value' in sales_summary:
                metrics_data.append(('Avg Transaction', f"${sales_summary['avg_transaction_value']:.2f}"))
            if 'total_days' in sales_summary:
                metrics_data.append(('Days Analyzed', f"{sales_summary['total_days']}"))
        
        for i, (metric_name, metric_value) in enumerate(metrics_data[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.6, metric_value, ha='center', va='center',
                   fontsize=24, fontweight='bold', color='darkblue')
            ax.text(0.5, 0.3, metric_name, ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False,
                                      edgecolor='steelblue', linewidth=2))
        
        # 2. Daily Revenue Trend
        ax = fig.add_subplot(gs[1, :2])
        
        if len(daily_revenue) > 0:
            # Convert Date to datetime if string
            if 'Date' in daily_revenue.columns:
                daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
                daily_revenue = daily_revenue.sort_values('Date')
            
            ax.plot(range(len(daily_revenue)), daily_revenue['total_revenue'],
                   linewidth=2, color='steelblue', marker='o', markersize=3)
            ax.fill_between(range(len(daily_revenue)), daily_revenue['total_revenue'],
                           alpha=0.3, color='lightblue')
            
            ax.set_xlabel('Day', fontsize=11)
            ax.set_ylabel('Revenue ($)', fontsize=11)
            ax.set_title('Daily Revenue Trend', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 3. Top Products
        ax = fig.add_subplot(gs[1, 2:])
        
        if len(coffee_counts) > 0:
            top_products = coffee_counts.nlargest(8, 'sales_count')
            
            colors = sns.color_palette('viridis', len(top_products))
            bars = ax.barh(range(len(top_products)), top_products['sales_count'], color=colors)
            
            ax.set_yticks(range(len(top_products)))
            ax.set_yticklabels(top_products['coffee_name'], fontsize=10)
            ax.set_xlabel('Transaction Count', fontsize=11)
            ax.set_title('Top 8 Products', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width):,}', ha='left', va='center', fontsize=9)
        
        # 4. Payment Distribution
        ax = fig.add_subplot(gs[2, 0])
        
        if len(payment_dist) > 0:
            colors = sns.color_palette('pastel', len(payment_dist))
            
            wedges, texts, autotexts = ax.pie(payment_dist['transaction_count'],
                                              labels=payment_dist['cash_type'],
                                              colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
            
            ax.set_title('Payment Methods', fontsize=13, fontweight='bold')
        
        # 5. Monthly Revenue
        ax = fig.add_subplot(gs[2, 1])
        
        if len(monthly_data) > 0:
            colors = sns.color_palette('coolwarm', len(monthly_data))
            bars = ax.bar(range(len(monthly_data)), monthly_data['total_revenue'],
                         color=colors)
            
            ax.set_xticks(range(len(monthly_data)))
            ax.set_xticklabels(monthly_data.get('Month_name', monthly_data.index),
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Revenue ($)', fontsize=11)
            ax.set_title('Monthly Revenue', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Hourly Pattern
        ax = fig.add_subplot(gs[2, 2:])
        
        hourly_data = self.load_csv_data('hourly_detailed_analysis.csv')
        
        if len(hourly_data) > 0:
            ax.bar(hourly_data['hour_of_day'], hourly_data['total_revenue'],
                  alpha=0.7, color='steelblue')
            
            ax2 = ax.twinx()
            ax2.plot(hourly_data['hour_of_day'], hourly_data['transaction_count'],
                    color='orange', marker='o', linewidth=2, label='Transactions')
            
            ax.set_xlabel('Hour of Day', fontsize=11)
            ax.set_ylabel('Revenue ($)', fontsize=11, color='steelblue')
            ax2.set_ylabel('Transactions', fontsize=11, color='orange')
            ax.set_title('Hourly Demand Pattern', fontsize=13, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='steelblue')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax.grid(True, alpha=0.3, axis='y')
        
        return self.save_figure(fig, 'dashboard_executive_summary.png')
    
    @log_execution_time(setup_logger(__name__))
    def create_interactive_dashboard(self) -> Optional[Path]:
        """
        Create interactive dashboard using Plotly.
        
        Returns:
            Path to saved HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available. Skipping interactive dashboard.")
            return None
        
        self.logger.info("Creating interactive dashboard...")
        
        # Load data
        sales_summary = self.load_json_data('sales_summary_metrics.json')
        daily_revenue = self.load_csv_data('daily_revenue.csv')
        hourly_data = self.load_csv_data('hourly_detailed_analysis.csv')
        coffee_counts = self.load_csv_data('coffee_counts.csv')
        payment_dist = self.load_csv_data('payment_distribution.csv')
        monthly_data = self.load_csv_data('monthly_trends_analysis.csv')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Daily Revenue Trend', 'Top Products',
                          'Hourly Demand Pattern', 'Payment Distribution',
                          'Monthly Revenue', 'Product Performance'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Daily Revenue Trend
        if len(daily_revenue) > 0:
            if 'Date' in daily_revenue.columns:
                daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
                daily_revenue = daily_revenue.sort_values('Date')
                x_data = daily_revenue['Date']
            else:
                x_data = range(len(daily_revenue))
            
            fig.add_trace(
                go.Scatter(x=x_data, y=daily_revenue['total_revenue'],
                          mode='lines+markers', name='Daily Revenue',
                          line=dict(color='steelblue', width=2)),
                row=1, col=1
            )
        
        # 2. Top Products
        if len(coffee_counts) > 0:
            top_products = coffee_counts.nlargest(10, 'sales_count')
            
            fig.add_trace(
                go.Bar(x=top_products['sales_count'], y=top_products['coffee_name'],
                      orientation='h', name='Transaction Count',
                      marker=dict(color=top_products['sales_count'],
                                colorscale='Viridis')),
                row=1, col=2
            )
        
        # 3. Hourly Demand Pattern
        if len(hourly_data) > 0:
            fig.add_trace(
                go.Scatter(x=hourly_data['hour_of_day'],
                          y=hourly_data['total_revenue'],
                          mode='lines+markers', name='Revenue',
                          line=dict(color='steelblue', width=2)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=hourly_data['hour_of_day'],
                          y=hourly_data['transaction_count'],
                          mode='lines+markers', name='Transactions',
                          line=dict(color='orange', width=2),
                          yaxis='y2'),
                row=2, col=1
            )
        
        # 4. Payment Distribution
        if len(payment_dist) > 0:
            fig.add_trace(
                go.Pie(labels=payment_dist['cash_type'],
                      values=payment_dist['transaction_count'],
                      name='Payment Methods'),
                row=2, col=2
            )
        
        # 5. Monthly Revenue
        if len(monthly_data) > 0:
            fig.add_trace(
                go.Bar(x=monthly_data.get('Month_name', monthly_data.index),
                      y=monthly_data['total_revenue'],
                      name='Monthly Revenue',
                      marker=dict(color=monthly_data['total_revenue'],
                                colorscale='RdYlGn')),
                row=3, col=1
            )
        
        # 6. Product Performance (Revenue vs Count)
        if len(coffee_counts) > 0:
            product_perf = self.load_csv_data('product_performance_detailed.csv')
            
            if len(product_perf) > 0:
                fig.add_trace(
                    go.Scatter(x=product_perf['transaction_count'],
                              y=product_perf['total_revenue'],
                              mode='markers', name='Products',
                              marker=dict(size=10, color=product_perf['avg_price'],
                                        colorscale='Plasma', showscale=True,
                                        colorbar=dict(title="Avg Price")),
                              text=product_perf['coffee_name'],
                              hovertemplate='<b>%{text}</b><br>' +
                                          'Transactions: %{x}<br>' +
                                          'Revenue: $%{y:,.0f}<extra></extra>'),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Coffee Shop Sales Analysis - Interactive Dashboard",
            title_font=dict(size=24, color='darkblue'),
            showlegend=False,
            height=1200,
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Product", row=1, col=2)
        
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
        
        fig.update_xaxes(title_text="Month", row=3, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=3, col=1)
        
        fig.update_xaxes(title_text="Transaction Count", row=3, col=2)
        fig.update_yaxes(title_text="Revenue ($)", row=3, col=2)
        
        # Save to HTML
        output_path = self.output_dir / 'dashboard_interactive.html'
        fig.write_html(str(output_path))
        
        self.logger.info(f"Interactive dashboard saved to: {output_path}")
        
        return output_path
    
    @log_execution_time(setup_logger(__name__))
    def generate_all_visualizations(self) -> Dict[str, List[Path]]:
        """
        Generate all visualizations from all modules.
        
        Returns:
            Dictionary mapping module names to list of saved file paths
        """
        self.logger.info("=" * 80)
        self.logger.info("GENERATING ALL VISUALIZATIONS")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Sales Performance
        try:
            self.logger.info("\n" + "="*80)
            results['sales'] = self.sales_viz.create_all_visualizations()
        except Exception as e:
            self.logger.error(f"Failed to create sales visualizations: {e}")
            results['sales'] = []
        
        # Time-Based Demand
        try:
            self.logger.info("\n" + "="*80)
            results['time'] = self.time_viz.create_all_visualizations()
        except Exception as e:
            self.logger.error(f"Failed to create time visualizations: {e}")
            results['time'] = []
        
        # Product Preference
        try:
            self.logger.info("\n" + "="*80)
            results['product'] = self.product_viz.create_all_visualizations()
        except Exception as e:
            self.logger.error(f"Failed to create product visualizations: {e}")
            results['product'] = []
        
        # Payment Behavior
        try:
            self.logger.info("\n" + "="*80)
            results['payment'] = self.payment_viz.create_all_visualizations()
        except Exception as e:
            self.logger.error(f"Failed to create payment visualizations: {e}")
            results['payment'] = []
        
        # Seasonality and Trends
        try:
            self.logger.info("\n" + "="*80)
            results['seasonality'] = self.seasonality_viz.create_all_visualizations()
        except Exception as e:
            self.logger.error(f"Failed to create seasonality visualizations: {e}")
            results['seasonality'] = []
        
        # Dashboards
        try:
            self.logger.info("\n" + "="*80)
            dashboard_files = []
            
            exec_summary = self.create_executive_summary()
            dashboard_files.append(exec_summary)
            
            interactive = self.create_interactive_dashboard()
            if interactive:
                dashboard_files.append(interactive)
            
            results['dashboard'] = dashboard_files
        except Exception as e:
            self.logger.error(f"Failed to create dashboards: {e}")
            results['dashboard'] = []
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("VISUALIZATION GENERATION COMPLETE")
        self.logger.info("="*80)
        
        total_files = sum(len(files) for files in results.values())
        
        self.logger.info(f"\nTotal visualizations created: {total_files}")
        for module, files in results.items():
            self.logger.info(f"  {module}: {len(files)} files")
        
        self.logger.info(f"\nAll visualizations saved to: {self.output_dir}")
        
        return results
    
    @log_execution_time(setup_logger(__name__))
    def create_visualization_index(self) -> Path:
        """
        Create an index/catalog of all visualizations.
        
        Returns:
            Path to saved index file
        """
        self.logger.info("Creating visualization index...")
        
        # Scan output directory for visualization files
        viz_files = list(self.output_dir.glob('*.png')) + list(self.output_dir.glob('*.html'))
        
        # Categorize files
        categories = {
            'Sales Performance': [],
            'Time-Based Demand': [],
            'Product Preference': [],
            'Payment Behavior': [],
            'Seasonality & Trends': [],
            'Dashboards': []
        }
        
        for file in viz_files:
            filename = file.name.lower()
            
            if 'sales' in filename and 'dashboard' not in filename:
                categories['Sales Performance'].append(file.name)
            elif 'time' in filename or 'hourly' in filename or 'peak' in filename:
                categories['Time-Based Demand'].append(file.name)
            elif 'product' in filename:
                categories['Product Preference'].append(file.name)
            elif 'payment' in filename:
                categories['Payment Behavior'].append(file.name)
            elif 'seasonality' in filename or 'monthly' in filename or 'growth' in filename:
                categories['Seasonality & Trends'].append(file.name)
            elif 'dashboard' in filename:
                categories['Dashboards'].append(file.name)
        
        # Create index content
        index_content = "# Coffee Shop Sales Analysis - Visualization Index\n\n"
        index_content += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        index_content += f"Total visualizations: {len(viz_files)}\n\n"
        index_content += "---\n\n"
        
        for category, files in categories.items():
            if files:
                index_content += f"## {category}\n\n"
                for file in sorted(files):
                    index_content += f"- {file}\n"
                index_content += "\n"
        
        # Save index
        index_path = self.output_dir / 'VISUALIZATION_INDEX.md'
        index_path.write_text(index_content, encoding='utf-8')
        
        self.logger.info(f"Visualization index saved to: {index_path}")
        
        return index_path


if __name__ == "__main__":
    """Run dashboard generation when module is executed directly."""
    dashboard = DashboardGenerator()
    results = dashboard.generate_all_visualizations()
    dashboard.create_visualization_index()
