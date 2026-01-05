"""Coffee Sales Analysis - Main Pipeline

This is the main entry point for the complete coffee sales analysis pipeline.
It orchestrates data preprocessing, enrichment, and analysis based on ARD requirements.

Usage:
    python main.py [--skip-preprocessing] [--skip-enrichment] [--skip-analysis]
    
    Options:
        --skip-preprocessing    Skip data preprocessing step
        --skip-enrichment      Skip data enrichment step
        --skip-analysis        Skip analysis step
        --analyses             Comma-separated list of analyses to run
                              Options: sales,time,product,payment,seasonality
                              Default: all

Examples:
    python main.py                                    # Run complete pipeline
    python main.py --skip-preprocessing              # Use existing preprocessed data
    python main.py --analyses sales,time             # Run only sales and time analyses
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor import DataPreprocessor
from src.data.enrichment import DataEnrichment
from src.analysis.analyzer import CoffeeSalesAnalyzer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class CoffeeSalesPipeline:
    """Main pipeline orchestrator for coffee sales analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the pipeline.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.enriched_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[dict] = None
    
    def print_banner(self):
        """Print pipeline banner."""
        banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              COFFEE SALES ANALYSIS - COMPLETE PIPELINE                     ║
║                                                                            ║
║  Based on: Analysis Requirements Document (ARD)                           ║
║  Version: 1.0.0                                                           ║
║  Date: January 5, 2026                                                    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        self.logger.info("=" * 80)
        self.logger.info("COFFEE SALES ANALYSIS PIPELINE STARTED")
        self.logger.info("=" * 80)
    
    def run_preprocessing(self, force: bool = False) -> pd.DataFrame:
        """
        Run data preprocessing step.
        
        Args:
            force: Force reprocessing even if preprocessed data exists
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 1: DATA PREPROCESSING")
        self.logger.info("=" * 80)
        
        # Check if preprocessed data already exists
        preprocessed_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'preprocessed_data.csv'
        
        if not force and preprocessed_path.exists():
            self.logger.info(f"Loading existing preprocessed data from {preprocessed_path}")
            self.preprocessed_data = pd.read_csv(preprocessed_path)
            
            # Convert Date column if exists
            if 'Date' in self.preprocessed_data.columns:
                self.preprocessed_data['Date'] = pd.to_datetime(self.preprocessed_data['Date'])
            
            self.logger.info(f"Loaded {len(self.preprocessed_data)} rows, {len(self.preprocessed_data.columns)} columns")
        else:
            # Run preprocessing
            self.logger.info("Running preprocessing pipeline...")
            preprocessor = DataPreprocessor()
            self.preprocessed_data = preprocessor.preprocess()
            
            # Save preprocessed data
            saved_path = preprocessor.save_processed_data()
            self.logger.info(f"Preprocessed data saved to {saved_path}")
            
            # Print summary
            summary = preprocessor.get_data_summary()
            self.logger.info("\nPreprocessing Summary:")
            self.logger.info(f"  Shape: {summary['shape']}")
            if 'date_range' in summary:
                self.logger.info(f"  Date Range: {summary['date_range'].get('start')} to {summary['date_range'].get('end')}")
                self.logger.info(f"  Total Days: {summary['date_range'].get('days')}")
        
        return self.preprocessed_data
    
    def run_enrichment(self, df: Optional[pd.DataFrame] = None, force: bool = False) -> pd.DataFrame:
        """
        Run data enrichment step.
        
        Args:
            df: Preprocessed DataFrame. If None, uses self.preprocessed_data
            force: Force re-enrichment even if enriched data exists
            
        Returns:
            Enriched DataFrame
        """
        if df is None:
            df = self.preprocessed_data
        
        if df is None:
            raise ValueError("No preprocessed data available. Run preprocessing first.")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 2: DATA ENRICHMENT")
        self.logger.info("=" * 80)
        
        # Check if enriched data already exists
        enriched_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'enriched_coffee_sales.csv'
        
        if not force and enriched_path.exists():
            self.logger.info(f"Loading existing enriched data from {enriched_path}")
            self.enriched_data = pd.read_csv(enriched_path)
            
            # Convert Date column if exists
            if 'Date' in self.enriched_data.columns:
                self.enriched_data['Date'] = pd.to_datetime(self.enriched_data['Date'])
            
            self.logger.info(f"Loaded {len(self.enriched_data)} rows, {len(self.enriched_data.columns)} columns")
        else:
            # Run enrichment
            self.logger.info("Running enrichment pipeline...")
            enricher = DataEnrichment()
            self.enriched_data = enricher.enrich(df)
            
            # Save enriched data
            saved_path = enricher.save_enriched_data()
            self.logger.info(f"Enriched data saved to {saved_path}")
            
            # Print summary
            feature_summary = enricher.get_feature_summary()
            self.logger.info("\nEnrichment Summary:")
            self.logger.info(f"  Total Features: {feature_summary['total_features']}")
            self.logger.info(f"  Features Added: {feature_summary['total_features'] - len(df.columns)}")
            
            for category, features in feature_summary['feature_categories'].items():
                if features:
                    self.logger.info(f"  {category.replace('_', ' ').title()}: {len(features)}")
        
        return self.enriched_data
    
    def run_analysis(
        self,
        df: Optional[pd.DataFrame] = None,
        analyses: Optional[List[str]] = None,
        save_outputs: bool = True
    ) -> dict:
        """
        Run analysis step.
        
        Args:
            df: Enriched DataFrame. If None, uses self.enriched_data
            analyses: List of analyses to run. If None, runs all.
            save_outputs: Whether to save analysis outputs
            
        Returns:
            Dictionary with analysis results
        """
        if df is None:
            df = self.enriched_data
        
        if df is None:
            raise ValueError("No enriched data available. Run enrichment first.")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 3: ANALYSIS")
        self.logger.info("=" * 80)
        
        # Initialize analyzer
        analyzer = CoffeeSalesAnalyzer()
        
        # Run analyses
        if analyses:
            self.logger.info(f"Running selected analyses: {', '.join(analyses)}")
        else:
            self.logger.info("Running all analyses")
        
        self.analysis_results = analyzer.run_all_analyses(
            df,
            save_outputs=save_outputs,
            analyses=analyses
        )
        
        # Generate and display insights
        self.logger.info("\n" + "=" * 80)
        self.logger.info("KEY BUSINESS INSIGHTS")
        self.logger.info("=" * 80)
        
        analyzer.print_insights()
        
        # Get summary statistics
        summary = analyzer.get_summary_statistics()
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"  Analyses Completed: {len(summary['analyses_run'])}")
        self.logger.info(f"  Total Insights: {summary['total_insights']}")
        
        if 'key_metrics' in summary:
            metrics = summary['key_metrics']
            self.logger.info("\n  Key Metrics:")
            if 'total_revenue' in metrics:
                self.logger.info(f"    Total Revenue: ${metrics['total_revenue']:,.2f}")
            if 'total_transactions' in metrics:
                self.logger.info(f"    Total Transactions: {metrics['total_transactions']:,}")
            if 'avg_transaction_value' in metrics:
                self.logger.info(f"    Avg Transaction Value: ${metrics['avg_transaction_value']:.2f}")
        
        return self.analysis_results
    
    def run_complete_pipeline(
        self,
        skip_preprocessing: bool = False,
        skip_enrichment: bool = False,
        skip_analysis: bool = False,
        analyses: Optional[List[str]] = None,
        force_reprocess: bool = False
    ):
        """
        Run the complete analysis pipeline.
        
        Args:
            skip_preprocessing: Skip preprocessing step
            skip_enrichment: Skip enrichment step
            skip_analysis: Skip analysis step
            analyses: List of specific analyses to run
            force_reprocess: Force reprocessing even if cached data exists
        """
        self.print_banner()
        
        try:
            # Step 1: Preprocessing
            if not skip_preprocessing:
                self.preprocessed_data = self.run_preprocessing(force=force_reprocess)
            else:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("STEP 1: PREPROCESSING - SKIPPED")
                self.logger.info("=" * 80)
                # Try to load existing preprocessed data
                preprocessed_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'preprocessed_data.csv'
                if preprocessed_path.exists():
                    self.preprocessed_data = pd.read_csv(preprocessed_path)
                    if 'Date' in self.preprocessed_data.columns:
                        self.preprocessed_data['Date'] = pd.to_datetime(self.preprocessed_data['Date'])
                    self.logger.info(f"Loaded existing preprocessed data: {self.preprocessed_data.shape}")
            
            # Step 2: Enrichment
            if not skip_enrichment:
                self.enriched_data = self.run_enrichment(force=force_reprocess)
            else:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("STEP 2: ENRICHMENT - SKIPPED")
                self.logger.info("=" * 80)
                # Try to load existing enriched data
                enriched_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'enriched_coffee_sales.csv'
                if enriched_path.exists():
                    self.enriched_data = pd.read_csv(enriched_path)
                    if 'Date' in self.enriched_data.columns:
                        self.enriched_data['Date'] = pd.to_datetime(self.enriched_data['Date'])
                    self.logger.info(f"Loaded existing enriched data: {self.enriched_data.shape}")
            
            # Step 3: Analysis
            if not skip_analysis:
                self.analysis_results = self.run_analysis(analyses=analyses)
            else:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("STEP 3: ANALYSIS - SKIPPED")
                self.logger.info("=" * 80)
            
            # Pipeline completion
            self.print_completion_summary()
            
        except Exception as e:
            self.logger.error(f"\n{'=' * 80}")
            self.logger.error("PIPELINE FAILED")
            self.logger.error(f"{'=' * 80}")
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            raise
    
    def print_completion_summary(self):
        """Print pipeline completion summary."""
        self.logger.info("\n" + "╔" + "=" * 78 + "╗")
        self.logger.info("║" + " " * 20 + "PIPELINE COMPLETED SUCCESSFULLY" + " " * 27 + "║")
        self.logger.info("╚" + "=" * 78 + "╝")
        
        self.logger.info("\n📊 OUTPUTS GENERATED:")
        self.logger.info("  ├─ Preprocessed Data: data/enriched/preprocessed_data.csv")
        self.logger.info("  ├─ Enriched Data: data/enriched/enriched_coffee_sales.csv")
        
        output_dir = Path(self.paths.get('output', {}).get('root', 'output'))
        if output_dir.exists():
            csv_files = list(output_dir.glob('*.csv'))
            json_files = list(output_dir.glob('*.json'))
            txt_files = list(output_dir.glob('*.txt'))
            
            self.logger.info(f"  ├─ Analysis CSVs: {len(csv_files)} files")
            self.logger.info(f"  ├─ Analysis JSONs: {len(json_files)} files")
            self.logger.info(f"  └─ Reports: {len(txt_files)} files")
        
        self.logger.info("\n📈 NEXT STEPS:")
        self.logger.info("  1. Review executive summary: output/executive_summary.txt")
        self.logger.info("  2. Explore detailed analysis files in output/ directory")
        self.logger.info("  3. Create visualizations based on analysis results")
        self.logger.info("  4. Generate business reports and presentations")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("For detailed insights, see output/executive_summary.txt")
        self.logger.info("=" * 80 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Coffee Sales Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run complete pipeline
  python main.py --skip-preprocessing              # Skip preprocessing
  python main.py --skip-enrichment                 # Skip enrichment
  python main.py --skip-analysis                   # Skip analysis
  python main.py --analyses sales,time             # Run specific analyses
  python main.py --force                           # Force reprocessing all data
        """
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step'
    )
    
    parser.add_argument(
        '--skip-enrichment',
        action='store_true',
        help='Skip data enrichment step'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis step'
    )
    
    parser.add_argument(
        '--analyses',
        type=str,
        help='Comma-separated list of analyses to run (sales,time,product,payment,seasonality)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if cached data exists'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Parse analyses list if provided
    analyses = None
    if args.analyses:
        analyses = [a.strip() for a in args.analyses.split(',')]
        valid_analyses = {'sales', 'time', 'product', 'payment', 'seasonality'}
        invalid = set(analyses) - valid_analyses
        if invalid:
            print(f"Error: Invalid analyses: {invalid}")
            print(f"Valid options: {', '.join(valid_analyses)}")
            sys.exit(1)
    
    # Create and run pipeline
    pipeline = CoffeeSalesPipeline()
    
    pipeline.run_complete_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        skip_enrichment=args.skip_enrichment,
        skip_analysis=args.skip_analysis,
        analyses=analyses,
        force_reprocess=args.force
    )


if __name__ == "__main__":
    main()
