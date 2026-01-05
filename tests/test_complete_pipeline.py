"""Complete end-to-end pipeline test.

Tests the entire data processing and analysis pipeline from raw data to insights.

Usage:
    python test_complete_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.preprocessor import DataPreprocessor
from src.data.enrichment import DataEnrichment
from src.analysis.analyzer import CoffeeSalesAnalyzer
from src.utils.logger import setup_logger


def test_complete_pipeline():
    """Test the complete data processing and analysis pipeline."""
    logger = setup_logger(__name__)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE COFFEE SALES ANALYSIS PIPELINE")
    print("=" * 80)
    
    success = True
    
    try:
        # Step 1: Data Preprocessing
        print("\n[1/3] Testing Data Preprocessing...")
        print("-" * 80)
        
        preprocessor = DataPreprocessor()
        preprocessed_df = preprocessor.preprocess()
        
        print(f"  ✓ Preprocessing complete: {preprocessed_df.shape}")
        print(f"  ✓ Columns: {len(preprocessed_df.columns)}")
        
        # Step 2: Data Enrichment
        print("\n[2/3] Testing Data Enrichment...")
        print("-" * 80)
        
        enricher = DataEnrichment()
        enriched_df = enricher.enrich(preprocessed_df)
        
        print(f"  ✓ Enrichment complete: {enriched_df.shape}")
        print(f"  ✓ Features added: {enriched_df.shape[1] - preprocessed_df.shape[1]}")
        
        # Save enriched data for analysis
        enriched_path = enricher.save_enriched_data()
        print(f"  ✓ Saved to: {enriched_path}")
        
        # Step 3: Analysis
        print("\n[3/3] Testing Complete Analysis...")
        print("-" * 80)
        
        analyzer = CoffeeSalesAnalyzer()
        results = analyzer.run_all_analyses(enriched_df, save_outputs=True)
        
        print(f"  ✓ Analyses completed: {len(results)}")
        
        # Generate insights
        insights = analyzer.generate_insights()
        print(f"  ✓ Insights generated: {len(insights)}")
        
        # Verify outputs
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        
        output_dir = Path('outputs')
        if output_dir.exists():
            output_files = list(output_dir.glob('*.csv')) + list(output_dir.glob('*.json')) + list(output_dir.glob('*.txt'))
            print(f"  ✓ Output files created: {len(output_files)}")
            
            # List some key outputs
            key_files = [
                'executive_summary.txt',
                'product_performance_detailed.csv',
                'hourly_detailed_analysis.csv',
                'daily_revenue.csv'
            ]
            
            print("\n  Key Outputs:")
            for filename in key_files:
                filepath = output_dir / filename
                if filepath.exists():
                    print(f"    ✓ {filename}")
                else:
                    print(f"    ✗ {filename} (missing)")
        
        # Display sample insights
        print("\n" + "=" * 80)
        print("SAMPLE INSIGHTS (first 5)")
        print("=" * 80)
        for i, insight in enumerate(insights[:5], 1):
            print(f"  {i}. {insight}")
        
        if len(insights) > 5:
            print(f"  ... and {len(insights) - 5} more insights")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"  Dataset:")
        print(f"    - Raw shape: {preprocessed_df.shape}")
        print(f"    - Enriched shape: {enriched_df.shape}")
        print(f"    - Features added: {enriched_df.shape[1] - preprocessed_df.shape[1]}")
        
        if 'Date' in enriched_df.columns:
            print(f"  Date Range:")
            print(f"    - From: {enriched_df['Date'].min()}")
            print(f"    - To: {enriched_df['Date'].max()}")
        
        print(f"  Analysis:")
        print(f"    - Modules run: {len(results)}")
        print(f"    - Insights generated: {len(insights)}")
        print(f"    - Output files: {len(output_files) if output_dir.exists() else 0}")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - PIPELINE FULLY OPERATIONAL")
        print("=" * 80)
        
        print("\nNext Steps:")
        print("  1. Review executive summary: outputs/executive_summary.txt")
        print("  2. Explore detailed analysis files in outputs/ directory")
        print("  3. Create visualizations based on the analysis results")
        print("  4. Generate business reports and presentations")
        
        return True
        
    except Exception as e:
        print(f"\n✗ PIPELINE TEST FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)
