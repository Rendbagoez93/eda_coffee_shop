"""Quick test script to verify the data processing modules.

This script performs a quick test of the preprocessing and enrichment pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.data.enrichment import DataEnrichment
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def test_modules():
    """Test all modules are working correctly."""
    logger = setup_logger(__name__)
    
    print("\n" + "=" * 80)
    print("TESTING COFFEE SALES DATA PROCESSING MODULES")
    print("=" * 80)
    
    # Test 1: Configuration Loading
    print("\n[1/4] Testing Configuration Loading...")
    try:
        config_loader = ConfigLoader()
        config = config_loader.config
        paths = config_loader.paths
        
        print(f"  ✓ Config loaded: {config.get('project', {}).get('name')}")
        print(f"  ✓ Paths loaded: {len(paths)} path categories")
        print("  ✓ Configuration loading successful")
    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        return False
    
    # Test 2: Data Preprocessing
    print("\n[2/4] Testing Data Preprocessing...")
    try:
        preprocessor = DataPreprocessor()
        
        # Load data
        df = preprocessor.load_data()
        print(f"  ✓ Data loaded: {df.shape}")
        
        # Validate data
        validation = preprocessor.validate_data(df)
        print(f"  ✓ Validation complete: {validation['total_rows']} rows, {validation['duplicates']} duplicates")
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        print(f"  ✓ Data cleaned: {df_clean.shape}")
        
        # Convert types
        df_converted = preprocessor.convert_data_types(df_clean)
        print(f"  ✓ Types converted: {len(df_converted.dtypes)} columns")
        
        # Add basic features
        df_processed = preprocessor.add_basic_features(df_converted)
        print(f"  ✓ Basic features added: {df_processed.shape[1]} columns")
        
        print("  ✓ Preprocessing successful")
    except Exception as e:
        print(f"  ✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Data Enrichment
    print("\n[3/4] Testing Data Enrichment...")
    try:
        enricher = DataEnrichment()
        
        # Test individual enrichment functions
        df_time = enricher.add_time_features(df_processed.copy())
        print(f"  ✓ Time features: {df_time.shape[1]} columns")
        
        df_revenue = enricher.add_revenue_features(df_processed.copy())
        print(f"  ✓ Revenue features: {df_revenue.shape[1]} columns")
        
        df_product = enricher.add_product_features(df_processed.copy())
        print(f"  ✓ Product features: {df_product.shape[1]} columns")
        
        df_payment = enricher.add_payment_features(df_processed.copy())
        print(f"  ✓ Payment features: {df_payment.shape[1]} columns")
        
        # Full enrichment
        df_enriched = enricher.enrich(df_processed)
        print(f"  ✓ Full enrichment: {df_enriched.shape[1]} columns")
        
        print("  ✓ Enrichment successful")
    except Exception as e:
        print(f"  ✗ Enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Data Summary
    print("\n[4/4] Testing Data Summary...")
    try:
        summary = preprocessor.get_data_summary(df_enriched)
        feature_summary = enricher.get_feature_summary(df_enriched)
        
        print(f"  ✓ Summary generated:")
        print(f"    - Shape: {summary['shape']}")
        print(f"    - Total features: {feature_summary['total_features']}")
        print(f"    - Categories: {len([k for k, v in feature_summary['feature_categories'].items() if v])}")
        
        print("  ✓ Summary generation successful")
    except Exception as e:
        print(f"  ✗ Summary generation failed: {e}")
        return False
    
    # Final Results
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print(f"\nDataset Information:")
    print(f"  Original columns: {df.shape[1]}")
    print(f"  Processed columns: {df_processed.shape[1]}")
    print(f"  Enriched columns: {df_enriched.shape[1]}")
    print(f"  Features added: {df_enriched.shape[1] - df.shape[1]}")
    print(f"  Total rows: {df_enriched.shape[0]}")
    
    if 'Date' in df_enriched.columns:
        print(f"\nDate Range:")
        print(f"  From: {df_enriched['Date'].min()}")
        print(f"  To: {df_enriched['Date'].max()}")
    
    print(f"\nModules are ready to use!")
    print(f"  Run: python -m src.data.pipeline")
    
    return True


if __name__ == "__main__":
    success = test_modules()
    sys.exit(0 if success else 1)
