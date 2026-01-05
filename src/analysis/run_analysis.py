"""Main analysis pipeline runner.

This script demonstrates how to run the complete analysis pipeline
on the enriched coffee sales dataset.

Usage:
    python -m src.analysis.run_analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.analysis.analyzer import CoffeeSalesAnalyzer
from src.utils.logger import setup_logger


def run_complete_analysis():
    """Execute the complete analysis pipeline."""
    logger = setup_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("COFFEE SALES ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Load enriched data
        enriched_data_path = "data/enriched/enriched_coffee_sales.csv"
        
        if not Path(enriched_data_path).exists():
            logger.error(f"Enriched data not found at {enriched_data_path}")
            logger.info("Please run the data preprocessing pipeline first:")
            logger.info("  python -m src.data.pipeline")
            return
        
        logger.info(f"Loading enriched data from {enriched_data_path}")
        df = pd.read_csv(enriched_data_path)
        
        # Convert date column if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Initialize analyzer
        logger.info("\nInitializing Coffee Sales Analyzer...")
        analyzer = CoffeeSalesAnalyzer()
        
        # Run all analyses
        logger.info("\nRunning all analyses...")
        results = analyzer.run_all_analyses(df, save_outputs=True)
        
        # Print insights
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE - KEY INSIGHTS")
        logger.info("=" * 80)
        analyzer.print_insights()
        
        # Get summary
        summary = analyzer.get_summary_statistics()
        logger.info("\nSummary Statistics:")
        logger.info(f"  Analyses Run: {len(summary['analyses_run'])}")
        logger.info(f"  Total Insights: {summary['total_insights']}")
        
        if 'key_metrics' in summary:
            metrics = summary['key_metrics']
            if 'total_revenue' in metrics:
                logger.info(f"  Total Revenue: ${metrics['total_revenue']:,.2f}")
            if 'total_transactions' in metrics:
                logger.info(f"  Total Transactions: {metrics['total_transactions']:,}")
            if 'avg_transaction_value' in metrics:
                logger.info(f"  Avg Transaction: ${metrics['avg_transaction_value']:.2f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nOutputs saved to 'outputs/' directory")
        logger.info("Executive summary: outputs/executive_summary.txt")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_complete_analysis()
