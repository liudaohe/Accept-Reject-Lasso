"""
Logging utilities for Lasso Feature Selector.

This module provides structured logging capabilities for tracking
feature selection processes and debugging.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

def setup_logger(name: str = "lasso_feature_selector",
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for log output
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class LassoLogger:
    """
    Specialized logger for Lasso Feature Selector operations.
    
    This class provides structured logging with timing capabilities
    and method-specific logging contexts.
    """
    
    def __init__(self, 
                 name: str = "lasso_feature_selector",
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 enable_timing: bool = True):
        """
        Initialize Lasso logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            enable_timing: Whether to enable timing functionality
        """
        self.logger = setup_logger(name, level, log_file)
        self.enable_timing = enable_timing
        self._timers = {}
        self._method_context = None
    
    def set_method_context(self, method_name: str) -> None:
        """Set the current method context for logging."""
        self._method_context = method_name
    
    def clear_method_context(self) -> None:
        """Clear the current method context."""
        self._method_context = None
    
    def _format_message(self, message: str) -> str:
        """Format message with method context if available."""
        if self._method_context:
            return f"[{self._method_context}] {message}"
        return message
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def start_timer(self, timer_name: str) -> None:
        """Start a named timer."""
        if self.enable_timing:
            self._timers[timer_name] = time.time()
            self.debug(f"Timer '{timer_name}' started")
    
    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            timer_name: Name of the timer to stop
            
        Returns:
            Elapsed time in seconds
        """
        if not self.enable_timing or timer_name not in self._timers:
            return 0.0
        
        elapsed = time.time() - self._timers[timer_name]
        del self._timers[timer_name]
        
        self.info(f"Timer '{timer_name}' completed in {elapsed:.2f} seconds")
        return elapsed
    
    def log_method_start(self, method_name: str, **params) -> None:
        """Log the start of a method execution."""
        self.set_method_context(method_name)
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.info(f"Starting {method_name} with parameters: {param_str}")
        self.start_timer(f"{method_name}_execution")
    
    def log_method_end(self, method_name: str, result_summary: Optional[str] = None) -> None:
        """Log the end of a method execution."""
        elapsed = self.stop_timer(f"{method_name}_execution")
        
        if result_summary:
            self.info(f"Completed {method_name} in {elapsed:.2f}s. {result_summary}")
        else:
            self.info(f"Completed {method_name} in {elapsed:.2f}s")
        
        self.clear_method_context()
    
    def log_feature_selection_result(self, method_name: str, 
                                   selected_features: set,
                                   total_features: int) -> None:
        """Log feature selection results."""
        selection_rate = len(selected_features) / total_features if total_features > 0 else 0
        self.info(f"{method_name} selected {len(selected_features)}/{total_features} features "
                 f"({selection_rate:.1%} selection rate)")
    
    def log_rescue_result(self, base_count: int, rescued_count: int, 
                         final_count: int) -> None:
        """Log feature rescue results."""
        self.info(f"Feature rescue: {base_count} base + {rescued_count} rescued = "
                 f"{final_count} total features")
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        metric_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items())
        self.info(f"Evaluation metrics - {metric_str}")
    
    def log_hyperparameters(self, method_name: str, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameter configuration."""
        self.debug(f"{method_name} hyperparameters: {hyperparams}")
    
    def log_data_info(self, X_shape: tuple, y_shape: tuple, 
                     feature_names: Optional[list] = None) -> None:
        """Log dataset information."""
        self.info(f"Dataset: X{X_shape}, y{y_shape}")
        if feature_names:
            self.debug(f"Features: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
    
    def log_problem_groups(self, groups: list, correlation_threshold: float) -> None:
        """Log problem group identification results."""
        group_sizes = [len(group) for group in groups]
        total_features = sum(group_sizes)
        
        self.info(f"Identified {len(groups)} problem groups with correlation > {correlation_threshold}")
        self.debug(f"Group sizes: {group_sizes}, Total features in groups: {total_features}")
    
    def log_clustering_result(self, n_clusters: int, silhouette_score: float,
                            cluster_sizes: list) -> None:
        """Log clustering analysis results."""
        self.info(f"Clustering: {n_clusters} clusters, silhouette score: {silhouette_score:.3f}")
        self.debug(f"Cluster sizes: {cluster_sizes}")
    
    def create_session_log(self, log_dir: str = "logs") -> str:
        """
        Create a session-specific log file.
        
        Args:
            log_dir: Directory to store log files
            
        Returns:
            Path to the created log file
        """
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"lasso_session_{timestamp}.log"
        
        # Add file handler to existing logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.logger.level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.info(f"Session log created: {log_file}")
        return str(log_file)
