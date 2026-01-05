"""Data processing pipeline runner.

This script demonstrates how to use the preprocessing and enrichment modules
to process the coffee sales dataset.

Usage:
    python -m src.data.pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.data.enrichment import DataEnrichment
from src.utils.logger import setup_logger


def run_pipeline():
    """Execute the complete data processing pipeline."""
    logger = setup_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("COFFEE SALES DATA PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Preprocessing
        logger.info("\nStep 1: Data Preprocessing")
        logger.info("-" * 80)
        
        preprocessor = DataPreprocessor()
        preprocessed_df = preprocessor.preprocess()
        
        # Save preprocessed data
        preprocessed_path = preprocessor.save_processed_data()
        logger.info(f"Preprocessed data saved to: {preprocessed_path}")
        
        # Get data summary
        summary = preprocessor.get_data_summary()
        logger.info(f"\nData Summary:")
        logger.info(f"  Shape: {summary['shape']}")
        logger.info(f"  Date Range: {summary.get('date_range', {}).get('start')} to {summary.get('date_range', {}).get('end')}")
        logger.info(f"  Total Days: {summary.get('date_range', {}).get('days')}")
        
        # Step 2: Enrichment
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Data Enrichment")
        logger.info("-" * 80)
        
        enricher = DataEnrichment()
        enriched_df = enricher.enrich(preprocessed_df)
        
        # Save enriched data
        enriched_path = enricher.save_enriched_data()
        logger.info(f"Enriched data saved to: {enriched_path}")
        
        # Get feature summary
        feature_summary = enricher.get_feature_summary()
        logger.info(f"\nFeature Summary:")
        logger.info(f"  Total Features: {feature_summary['total_features']}")
        for category, features in feature_summary['feature_categories'].items():
            if features:
                logger.info(f"  {category.replace('_', ' ').title()}: {len(features)}")
        
        # Pipeline completion
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nOutputs:")
        logger.info(f"  1. Preprocessed: {preprocessed_path}")
        logger.info(f"  2. Enriched: {enriched_path}")
        logger.info(f"\nNext Steps:")
        logger.info(f"  - Run analysis modules in src/analysis/")
        logger.info(f"  - Generate visualizations")
        logger.info(f"  - Create business reports")
        
        return enriched_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()
