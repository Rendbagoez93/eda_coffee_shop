"""
Test script for visualization modules.

This script tests individual visualization modules and verifies outputs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.visualization import (
    SalesVisualizer,
    TimeVisualizer,
    ProductVisualizer,
    PaymentVisualizer,
    SeasonalityVisualizer,
    DashboardGenerator
)
from src.utils.logger import setup_logger


def test_sales_visualizations():
    """Test sales performance visualizations."""
    print("\n" + "="*80)
    print("Testing Sales Visualizations")
    print("="*80)
    
    viz = SalesVisualizer()
    
    try:
        # Test individual plots
        print("Testing revenue overview...")
        file1 = viz.plot_revenue_overview()
        print(f"✓ Created: {file1}")
        
        print("Testing weekday analysis...")
        file2 = viz.plot_weekday_analysis()
        print(f"✓ Created: {file2}")
        
        print("Testing price analysis...")
        file3 = viz.plot_price_analysis()
        print(f"✓ Created: {file3}")
        
        print("\n✓ Sales visualizations: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Sales visualizations: FAILED - {e}")
        return False


def test_time_visualizations():
    """Test time-based demand visualizations."""
    print("\n" + "="*80)
    print("Testing Time Visualizations")
    print("="*80)
    
    viz = TimeVisualizer()
    
    try:
        print("Testing hourly demand...")
        file1 = viz.plot_hourly_demand()
        print(f"✓ Created: {file1}")
        
        print("Testing heatmap...")
        file2 = viz.plot_heatmap()
        print(f"✓ Created: {file2}")
        
        print("Testing peak analysis...")
        file3 = viz.plot_peak_analysis()
        print(f"✓ Created: {file3}")
        
        print("\n✓ Time visualizations: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Time visualizations: FAILED - {e}")
        return False


def test_product_visualizations():
    """Test product preference visualizations."""
    print("\n" + "="*80)
    print("Testing Product Visualizations")
    print("="*80)
    
    viz = ProductVisualizer()
    
    try:
        print("Testing product popularity...")
        file1 = viz.plot_product_popularity()
        print(f"✓ Created: {file1}")
        
        print("Testing time-of-day patterns...")
        file2 = viz.plot_time_of_day_patterns()
        print(f"✓ Created: {file2}")
        
        print("Testing hourly patterns...")
        file3 = viz.plot_hourly_patterns()
        print(f"✓ Created: {file3}")
        
        print("Testing product metrics...")
        file4 = viz.plot_product_metrics()
        print(f"✓ Created: {file4}")
        
        print("\n✓ Product visualizations: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Product visualizations: FAILED - {e}")
        return False


def test_payment_visualizations():
    """Test payment behavior visualizations."""
    print("\n" + "="*80)
    print("Testing Payment Visualizations")
    print("="*80)
    
    viz = PaymentVisualizer()
    
    try:
        print("Testing payment distribution...")
        file1 = viz.plot_payment_distribution()
        print(f"✓ Created: {file1}")
        
        print("Testing spending patterns...")
        file2 = viz.plot_spending_patterns()
        print(f"✓ Created: {file2}")
        
        print("Testing payment trends...")
        file3 = viz.plot_payment_trends()
        print(f"✓ Created: {file3}")
        
        print("Testing payment-product relationship...")
        file4 = viz.plot_payment_product_relationship()
        print(f"✓ Created: {file4}")
        
        print("\n✓ Payment visualizations: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Payment visualizations: FAILED - {e}")
        return False


def test_seasonality_visualizations():
    """Test seasonality and trend visualizations."""
    print("\n" + "="*80)
    print("Testing Seasonality Visualizations")
    print("="*80)
    
    viz = SeasonalityVisualizer()
    
    try:
        print("Testing monthly trends...")
        file1 = viz.plot_monthly_trends()
        print(f"✓ Created: {file1}")
        
        print("Testing seasonal patterns...")
        file2 = viz.plot_seasonal_patterns()
        print(f"✓ Created: {file2}")
        
        print("Testing growth analysis...")
        file3 = viz.plot_growth_analysis()
        print(f"✓ Created: {file3}")
        
        print("\n✓ Seasonality visualizations: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Seasonality visualizations: FAILED - {e}")
        return False


def test_dashboard():
    """Test dashboard generation."""
    print("\n" + "="*80)
    print("Testing Dashboard Generation")
    print("="*80)
    
    dashboard = DashboardGenerator()
    
    try:
        print("Testing executive summary...")
        file1 = dashboard.create_executive_summary()
        print(f"✓ Created: {file1}")
        
        print("Testing interactive dashboard...")
        file2 = dashboard.create_interactive_dashboard()
        if file2:
            print(f"✓ Created: {file2}")
        else:
            print("⚠ Interactive dashboard skipped (Plotly not available)")
        
        print("Testing visualization index...")
        file3 = dashboard.create_visualization_index()
        print(f"✓ Created: {file3}")
        
        print("\n✓ Dashboard generation: PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Dashboard generation: FAILED - {e}")
        return False


def test_all():
    """Run all visualization tests."""
    logger = setup_logger(__name__)
    
    print("="*80)
    print("VISUALIZATION MODULE TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test each module
    results['sales'] = test_sales_visualizations()
    results['time'] = test_time_visualizations()
    results['product'] = test_product_visualizations()
    results['payment'] = test_payment_visualizations()
    results['seasonality'] = test_seasonality_visualizations()
    results['dashboard'] = test_dashboard()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for module, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{module.ljust(20)}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Check output directory
    output_dir = Path("output")
    if output_dir.exists():
        png_files = list(output_dir.glob("*.png"))
        html_files = list(output_dir.glob("*.html"))
        
        print(f"\nGenerated files:")
        print(f"  PNG images: {len(png_files)}")
        print(f"  HTML files: {len(html_files)}")
        print(f"  Total: {len(png_files) + len(html_files)}")
    
    if passed == total:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print(f"✗ {total - passed} TEST(S) FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(test_all())
