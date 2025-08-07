"""
Interface definitions for the Lasso Feature Selector package.

This module defines the core interfaces that all feature selection methods
must implement to ensure consistency and interoperability.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from .types import DataFrame, Series, FeatureSet, Metadata, SelectionResult

@runtime_checkable
class GlobalLassoProvider(Protocol):
    """
    Protocol for global Lasso feature selection providers.
    
    Global providers operate on the entire dataset and return both
    selected features and metadata for use by subset providers.
    """
    
    def __call__(self, X: DataFrame, y: Series) -> SelectionResult:
        """
        Perform global feature selection.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            Tuple of (selected_features, metadata)
        """
        ...

@runtime_checkable  
class SubsetLassoProvider(Protocol):
    """
    Protocol for subset Lasso feature selection providers.
    
    Subset providers operate on data subsets using metadata from
    the global provider to maintain consistency.
    """
    
    def __call__(self, X_subset: DataFrame, y_subset: Series, metadata: Metadata) -> FeatureSet:
        """
        Perform subset feature selection.
        
        Args:
            X_subset: Subset feature matrix
            y_subset: Subset target vector  
            metadata: Metadata from global provider
            
        Returns:
            Set of selected feature names
        """
        ...

class LassoMethodInterface(ABC):
    """
    Abstract base class for all Lasso-based feature selection methods.
    
    This interface ensures that all methods provide both global and subset
    implementations with consistent signatures.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the method name."""
        pass
    
    @property  
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the method."""
        pass
    
    @abstractmethod
    def global_provider(self) -> GlobalLassoProvider:
        """Return the global feature selection provider."""
        pass
    
    @abstractmethod
    def subset_provider(self) -> SubsetLassoProvider:
        """Return the subset feature selection provider."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return the current hyperparameter configuration."""
        pass
    
    @abstractmethod
    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameter configuration."""
        pass

class RescueMethodInterface(ABC):
    """
    Abstract base class for feature rescue methods.
    
    Rescue methods identify and recover features that may have been
    missed by the primary selection process.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the rescue method name."""
        pass
    
    @abstractmethod
    def identify_problem_groups(self, X: DataFrame, **kwargs) -> list:
        """Identify groups of problematic (highly correlated) features."""
        pass
    
    @abstractmethod
    def rescue_features(self, 
                       X: DataFrame, 
                       y: Series,
                       base_features: FeatureSet,
                       problem_groups: list,
                       **kwargs) -> FeatureSet:
        """Rescue additional features based on problem group analysis."""
        pass
