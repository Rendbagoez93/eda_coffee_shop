"""Configuration loader utility for YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages project configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Dictionary containing configuration
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value.
        
        Args:
            config_name: Name of the config file
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config(config_name)
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory.
        
        Returns:
            Dictionary with all configurations
        """
        config_files = ['config', 'paths', 'logging']
        
        for config_name in config_files:
            try:
                self.load_config(config_name)
            except FileNotFoundError:
                continue
        
        return self._configs
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.load_config('paths')
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get main configuration."""
        return self.load_config('config')
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.load_config('logging')
