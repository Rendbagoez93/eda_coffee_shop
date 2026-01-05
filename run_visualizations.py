"""
Visualization Runner for Coffee Shop Sales Analysis

This script generates all visualizations and dashboards based on the analysis results.

Usage:
    python run_visualizations.py [options]

Options:
    --module MODULES    Generate only specific modules (comma-separated): sales,time,product,payment,seasonality,dashboard
    --skip-dashboard    Skip dashboard generation
    --output-dir PATH   Custom output directory (default: from config)

Examples:
    python run_visualizations.py
    python run_visualizations.py --module sales,time
    python run_visualizations.py --skip-dashboard
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.visualization.dashboard import DashboardGenerator
from src.visualization.sales_viz import SalesVisualizer
from src.visualization.time_viz import TimeVisualizer
from src.visualization.product_viz import ProductVisualizer
from src.visualization.payment_viz import PaymentVisualizer
from src.visualization.seasonality_viz import SeasonalityVisualizer
from src.utils.logger import setup_logger


def main():
    """Main visualization generation function."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Generate visualizations for coffee shop sales analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--module',
        type=str,
        help='Generate only specific modules (comma-separated): sales,time,product,payment,seasonality,dashboard'
    )
    
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard generation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("COFFEE SHOP SALES ANALYSIS - VISUALIZATION GENERATION")
    logger.info("=" * 80)
    
    # Determine which modules to run
    if args.module:
        modules_to_run = [m.strip().lower() for m in args.module.split(',')]
        logger.info(f"Running specific modules: {', '.join(modules_to_run)}")
    else:
        modules_to_run = ['sales', 'time', 'product', 'payment', 'seasonality']
        if not args.skip_dashboard:
            modules_to_run.append('dashboard')
        logger.info("Running all visualization modules")
    
    # Initialize visualizers
    results = {}
    
    try:
        # Sales Performance
        if 'sales' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("SALES PERFORMANCE VISUALIZATIONS")
            logger.info("="*80)
            
            sales_viz = SalesVisualizer()
            results['sales'] = sales_viz.create_all_visualizations()
            logger.info(f"Created {len(results['sales'])} sales visualizations")
        
        # Time-Based Demand
        if 'time' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("TIME-BASED DEMAND VISUALIZATIONS")
            logger.info("="*80)
            
            time_viz = TimeVisualizer()
            results['time'] = time_viz.create_all_visualizations()
            logger.info(f"Created {len(results['time'])} time visualizations")
        
        # Product Preference
        if 'product' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("PRODUCT PREFERENCE VISUALIZATIONS")
            logger.info("="*80)
            
            product_viz = ProductVisualizer()
            results['product'] = product_viz.create_all_visualizations()
            logger.info(f"Created {len(results['product'])} product visualizations")
        
        # Payment Behavior
        if 'payment' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("PAYMENT BEHAVIOR VISUALIZATIONS")
            logger.info("="*80)
            
            payment_viz = PaymentVisualizer()
            results['payment'] = payment_viz.create_all_visualizations()
            logger.info(f"Created {len(results['payment'])} payment visualizations")
        
        # Seasonality and Trends
        if 'seasonality' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("SEASONALITY & TRENDS VISUALIZATIONS")
            logger.info("="*80)
            
            seasonality_viz = SeasonalityVisualizer()
            results['seasonality'] = seasonality_viz.create_all_visualizations()
            logger.info(f"Created {len(results['seasonality'])} seasonality visualizations")
        
        # Dashboards
        if 'dashboard' in modules_to_run:
            logger.info("\n" + "="*80)
            logger.info("DASHBOARD GENERATION")
            logger.info("="*80)
            
            dashboard_gen = DashboardGenerator()
            
            # Executive summary
            exec_summary = dashboard_gen.create_executive_summary()
            logger.info(f"Executive summary created: {exec_summary}")
            
            # Interactive dashboard
            interactive = dashboard_gen.create_interactive_dashboard()
            if interactive:
                logger.info(f"Interactive dashboard created: {interactive}")
            
            # Visualization index
            index = dashboard_gen.create_visualization_index()
            logger.info(f"Visualization index created: {index}")
            
            results['dashboard'] = [exec_summary]
            if interactive:
                results['dashboard'].append(interactive)
            results['dashboard'].append(index)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("VISUALIZATION GENERATION COMPLETE")
        logger.info("="*80)
        
        total_files = sum(len(files) for files in results.values())
        logger.info(f"\nTotal visualizations created: {total_files}")
        
        for module, files in results.items():
            logger.info(f"  {module}: {len(files)} files")
            for file in files:
                logger.info(f"    - {file.name}")
        
        logger.info("\nAll visualizations saved to: output/")
        logger.info("\n" + "="*80)
        logger.info("SUCCESS: All visualizations generated successfully!")
        logger.info("="*80)
        
        return 0
    
    except Exception as e:
        logger.error(f"\nERROR: Visualization generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
