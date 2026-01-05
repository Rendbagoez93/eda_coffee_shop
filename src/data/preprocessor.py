"""Data preprocessing module for Coffee Sales Analysis.

This module handles:
- Data loading from raw CSV
- Data validation and quality checks
- Data cleaning (missing values, duplicates)
- Data type conversions
- Basic data transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger, log_dataframe_info, log_execution_time


class DataPreprocessor:
    """Preprocesses raw coffee sales data for analysis."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize DataPreprocessor.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.config = self.config_loader.config
        self.paths = self.config_loader.paths
        self.logger = setup_logger(__name__)
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.quality_report: Dict = {}
    
    @log_execution_time(setup_logger(__name__))
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw coffee sales data.
        
        Args:
            file_path: Path to CSV file. If None, uses config path.
            
        Returns:
            Raw DataFrame
        """
        if file_path is None:
            file_path = self.paths.get('raw_data', {}).get('coffee_sales')
        
        if not file_path:
            raise ValueError("No data file path provided")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from {file_path}")
        
        # Load with appropriate encoding
        encoding = self.config.get('data_processing', {}).get('encoding', 'utf-8')
        
        try:
            self.raw_data = pd.read_csv(file_path, encoding=encoding)
            log_dataframe_info(self.logger, self.raw_data, "Raw data")
            
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def validate_data(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Validate data quality and structure.
        
        Args:
            df: DataFrame to validate. If None, uses raw_data.
            
        Returns:
            Dictionary with validation results
        """
        if df is None:
            df = self.raw_data
        
        if df is None:
            raise ValueError("No data to validate")
        
        self.logger.info("Validating data quality...")
        
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'duplicates': df.duplicated().sum(),
            'missing_values': {},
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        validation_report['missing_values'] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Check for expected columns (from ARD.md)
        expected_columns = [
            'hour_of_day', 'cash_type', 'money', 'coffee_name',
            'Time_of_Day', 'Weekday', 'Weekdaysort', 'Month_name',
            'Monthsort', 'Date', 'Time'
        ]
        
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            validation_report['missing_columns'] = list(missing_columns)
            self.logger.warning(f"Missing expected columns: {missing_columns}")
        
        # Log validation results
        self.logger.info(f"Validation complete - Rows: {validation_report['total_rows']}, "
                        f"Duplicates: {validation_report['duplicates']}, "
                        f"Missing values: {sum(validation_report['missing_values'].values())}")
        
        self.quality_report = validation_report
        return validation_report
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the data by handling missing values and duplicates.
        
        Args:
            df: DataFrame to clean. If None, uses raw_data.
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.raw_data.copy()
        else:
            df = df.copy()
        
        if df is None:
            raise ValueError("No data to clean")
        
        self.logger.info("Cleaning data...")
        initial_rows = len(df)
        
        # Handle duplicates
        if self.config.get('data_processing', {}).get('validation', {}).get('check_duplicates', True):
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                self.logger.info(f"Removing {duplicates} duplicate rows")
                df = df.drop_duplicates()
        
        # Handle missing values
        missing_strategy = self.config.get('data_processing', {}).get('missing_values', {}).get('strategy', 'drop')
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f"Found {missing_count} missing values. Strategy: {missing_strategy}")
            
            if missing_strategy == 'drop':
                df = df.dropna()
            elif missing_strategy == 'fill':
                fill_value = self.config.get('data_processing', {}).get('missing_values', {}).get('fill_value', 0)
                df = df.fillna(fill_value)
        
        rows_removed = initial_rows - len(df)
        self.logger.info(f"Cleaning complete - Removed {rows_removed} rows")
        
        return df
    
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            DataFrame with converted types
        """
        self.logger.info("Converting data types...")
        df = df.copy()
        
        # Convert numeric columns
        numeric_columns = ['hour_of_day', 'money', 'Weekdaysort', 'Monthsort']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert datetime columns
        date_format = self.config.get('data_processing', {}).get('date_format', '%Y-%m-%d')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
        
        # Convert time column if exists
        if 'Time' in df.columns:
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
            except:
                self.logger.warning("Could not convert Time column")
        
        # Convert categorical columns to category type for memory efficiency
        categorical_columns = ['cash_type', 'coffee_name', 'Time_of_Day', 'Weekday', 'Month_name']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        self.logger.info("Data type conversion complete")
        log_dataframe_info(self.logger, df, "Converted data")
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic derived features for preprocessing.
        
        Args:
            df: DataFrame to enhance
            
        Returns:
            DataFrame with basic features
        """
        self.logger.info("Adding basic features...")
        df = df.copy()
        
        # Add datetime column combining Date and Time if not exists
        if 'Date' in df.columns and 'Time' in df.columns and 'datetime' not in df.columns:
            try:
                # Convert Time to string if it's not already
                time_str = df['Time'].astype(str)
                df['datetime'] = pd.to_datetime(
                    df['Date'].astype(str) + ' ' + time_str,
                    errors='coerce'
                )
            except Exception as e:
                self.logger.warning(f"Could not create datetime column: {e}")
        
        # Add year, month, day columns if Date exists
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['day_of_week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
            df['week_of_year'] = df['Date'].dt.isocalendar().week
        
        # Add is_weekend flag
        if 'day_of_week' in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        self.logger.info(f"Added basic features - Total columns: {len(df.columns)}")
        
        return df
    
    @log_execution_time(setup_logger(__name__))
    def preprocess(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline.
        
        Args:
            file_path: Path to raw data file
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting data preprocessing pipeline")
        self.logger.info("=" * 80)
        
        # Step 1: Load data
        df = self.load_data(file_path)
        
        # Step 2: Validate data
        self.validate_data(df)
        
        # Step 3: Clean data
        df = self.clean_data(df)
        
        # Step 4: Convert data types
        df = self.convert_data_types(df)
        
        # Step 5: Add basic features
        df = self.add_basic_features(df)
        
        self.processed_data = df
        
        self.logger.info("=" * 80)
        self.logger.info("Preprocessing pipeline complete")
        self.logger.info("=" * 80)
        
        return df
    
    def save_processed_data(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Save preprocessed data to CSV.
        
        Args:
            df: DataFrame to save. If None, uses processed_data.
            output_path: Output file path. If None, uses config path.
            
        Returns:
            Path to saved file
        """
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No processed data to save")
        
        if output_path is None:
            output_path = Path(self.paths.get('data', {}).get('enriched', 'data/enriched')) / 'preprocessed_data.csv'
        else:
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving preprocessed data to {output_path}")
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Successfully saved {len(df)} rows to {output_path}")
        
        return output_path
    
    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get summary statistics of the data.
        
        Args:
            df: DataFrame to summarize. If None, uses processed_data.
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No data to summarize")
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {},
            'date_range': {}
        }
        
        # Categorical summaries
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        # Date range
        if 'Date' in df.columns:
            summary['date_range'] = {
                'start': str(df['Date'].min()),
                'end': str(df['Date'].max()),
                'days': (df['Date'].max() - df['Date'].min()).days
            }
        
        return summary
