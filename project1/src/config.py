"""
Configuration loader for the Airbnb Sentiment Analysis project.

This module handles loading and managing configuration from YAML files
and provides easy access to configuration parameters throughout the application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key (str): Configuration key (e.g., 'data.sample_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key (str): Configuration key (e.g., 'data.sample_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path (str, optional): Path to save configuration. Defaults to original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self._config.get('data', {})
    
    @property
    def analysis(self) -> Dict[str, Any]:
        """Get analysis configuration section."""
        return self._config.get('analysis', {})
    
    @property
    def forecasting(self) -> Dict[str, Any]:
        """Get forecasting configuration section."""
        return self._config.get('forecasting', {})
    
    @property
    def visualization(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self._config.get('visualization', {})
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration section."""
        return self._config.get('output', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self._config.get('logging', {})

# Global configuration instance
config = Config() 