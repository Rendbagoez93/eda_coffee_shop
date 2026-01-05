"""Logging utility for Coffee Sales Analysis."""

import logging
import logging.config
from pathlib import Path
from typing import Optional
import yaml


def setup_logger(
    name: str,
    config_path: Optional[str] = "config/logging.yaml",
    default_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with configuration from YAML file.
    
    Args:
        name: Logger name (typically __name__ of the module)
        config_path: Path to logging configuration file
        default_level: Default logging level if config file not found
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger already configured, return it
    if logger.handlers:
        return logger
    
    config_file = Path(config_path) if config_path else None
    
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # Ensure output directory exists for file handlers
                output_dir = Path('outputs')
                output_dir.mkdir(exist_ok=True)
                
                logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic config if loading fails
            logging.basicConfig(
                level=default_level,
                format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            logger = logging.getLogger(name)
            logger.warning(f"Failed to load logging config: {e}. Using basic config.")
    else:
        # Use basic configuration if no config file
        logging.basicConfig(
            level=default_level,
            format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(name)
    
    return logger


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """
    Log DataFrame information for debugging.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to identify the DataFrame in logs
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.debug(f"{name} columns: {list(df.columns)}")
    logger.debug(f"{name} dtypes:\n{df.dtypes}")
    logger.debug(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}...")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {elapsed_time:.2f}s")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed_time:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator
