"""
Configuration management system for Lasso Feature Selector.

This module provides a centralized configuration class that handles
parameter validation, default loading, and configuration persistence.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

from .defaults import DEFAULT_CONFIGS, HYPERPARAMETER_RANGES

class LassoConfig:
    """
    Centralized configuration management for Lasso Feature Selector.
    
    This class handles loading, validation, and management of hyperparameters
    for all feature selection methods and rescue procedures.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dict: Optional dictionary of configuration overrides
        """
        self._config = deepcopy(DEFAULT_CONFIGS)
        if config_dict:
            self.update(config_dict)
    
    def get(self, method: str, parameter: Optional[str] = None) -> Any:
        """
        Get configuration for a specific method or parameter.
        
        Args:
            method: Method name (e.g., 'adaptive_lasso', 'rescue')
            parameter: Specific parameter name (optional)
            
        Returns:
            Configuration value or dictionary
        """
        if method not in self._config:
            raise ValueError(f"Unknown method: {method}")
        
        method_config = self._config[method]
        
        if parameter is None:
            return deepcopy(method_config)
        
        if parameter not in method_config:
            raise ValueError(f"Parameter '{parameter}' not found for method '{method}'")
        
        return method_config[parameter]
    
    def set(self, method: str, parameter: str, value: Any) -> None:
        """
        Set a specific parameter for a method.
        
        Args:
            method: Method name
            parameter: Parameter name
            value: Parameter value
        """
        if method not in self._config:
            raise ValueError(f"Unknown method: {method}")
        
        # Validate parameter
        self._validate_parameter(parameter, value)
        
        self._config[method][parameter] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary of values.
        
        Args:
            config_dict: Dictionary with method->parameter->value structure
        """
        for method, params in config_dict.items():
            if method not in self._config:
                warnings.warn(f"Unknown method '{method}' in config, skipping")
                continue
            
            if not isinstance(params, dict):
                warnings.warn(f"Invalid config format for method '{method}', skipping")
                continue
            
            for param, value in params.items():
                try:
                    self.set(method, param, value)
                except ValueError as e:
                    warnings.warn(f"Invalid parameter '{param}' for method '{method}': {e}")
    
    def _validate_parameter(self, parameter: str, value: Any) -> None:
        """
        Validate a parameter value against defined ranges.
        
        Args:
            parameter: Parameter name
            value: Parameter value
            
        Raises:
            ValueError: If parameter value is invalid
        """
        if parameter not in HYPERPARAMETER_RANGES:
            # Allow unknown parameters with warning
            warnings.warn(f"Unknown parameter '{parameter}', skipping validation")
            return
        
        constraints = HYPERPARAMETER_RANGES[parameter]
        
        # Type validation
        expected_type = constraints['type']
        if not isinstance(value, expected_type):
            raise ValueError(f"Parameter '{parameter}' must be of type {expected_type.__name__}")
        
        # Range validation for numeric types
        if expected_type in (int, float):
            if 'min' in constraints and constraints['min'] is not None:
                if value < constraints['min']:
                    raise ValueError(f"Parameter '{parameter}' must be >= {constraints['min']}")
            
            if 'max' in constraints and constraints['max'] is not None:
                if value > constraints['max']:
                    raise ValueError(f"Parameter '{parameter}' must be <= {constraints['max']}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Path to save configuration file
        """
        filepath = Path(filepath)
        
        # Convert numpy arrays to lists for JSON serialization
        config_to_save = self._prepare_for_serialization(self._config)
        
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to configuration file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
        
        self.update(loaded_config)
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """
        Prepare configuration object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    def get_method_names(self) -> list:
        """Get list of available method names."""
        return list(self._config.keys())
    
    def reset_to_defaults(self, method: Optional[str] = None) -> None:
        """
        Reset configuration to defaults.
        
        Args:
            method: Specific method to reset (None for all methods)
        """
        if method is None:
            self._config = deepcopy(DEFAULT_CONFIGS)
        else:
            if method not in DEFAULT_CONFIGS:
                raise ValueError(f"Unknown method: {method}")
            self._config[method] = deepcopy(DEFAULT_CONFIGS[method])
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"LassoConfig({list(self._config.keys())})"
