"""
Algorithm manager for pixelation algorithms.

This module provides a centralized way to register, discover,
and instantiate pixelation algorithms.
"""

from typing import Dict, Type, List, Optional
from .base import PixelationAlgorithm
from .nearest_neighbor import (
    NearestNeighborAlgorithm, 
    BilinearPixelationAlgorithm, 
    LanczosPixelationAlgorithm
)
from .advanced import (
    EdgePreservingAlgorithm,
    SuperPixelAlgorithm,
    AdaptivePixelationAlgorithm,
    QuantizationAlgorithm,
    VoronoiAlgorithm,
    HexagonalAlgorithm
)


class AlgorithmManager:
    """Manages registration and instantiation of pixelation algorithms."""
    
    _algorithms: Dict[str, Type[PixelationAlgorithm]] = {}
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[PixelationAlgorithm]) -> None:
        """
        Register a pixelation algorithm.
        
        Args:
            name: Name to register the algorithm under
            algorithm_class: Algorithm class to register
        """
        cls._algorithms[name.lower()] = algorithm_class
    
    @classmethod
    def get_algorithm(cls, name: str) -> Optional[PixelationAlgorithm]:
        """
        Get an instance of a registered algorithm.
        
        Args:
            name: Name of the algorithm
            
        Returns:
            Algorithm instance or None if not found
        """
        algorithm_class = cls._algorithms.get(name.lower())
        if algorithm_class:
            return algorithm_class()
        return None
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """
        Get list of available algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_algorithm_info(cls) -> Dict[str, str]:
        """
        Get information about all registered algorithms.
        
        Returns:
            Dictionary mapping algorithm names to descriptions
        """
        info = {}
        for name, algorithm_class in cls._algorithms.items():
            instance = algorithm_class()
            info[name] = getattr(instance, 'description', instance.name)
        return info


# Register built-in algorithms
def register_builtin_algorithms():
    """Register all built-in algorithms."""
    AlgorithmManager.register_algorithm("nearest", NearestNeighborAlgorithm)
    AlgorithmManager.register_algorithm("bilinear", BilinearPixelationAlgorithm)
    AlgorithmManager.register_algorithm("lanczos", LanczosPixelationAlgorithm)
    AlgorithmManager.register_algorithm("edge_preserving", EdgePreservingAlgorithm)
    AlgorithmManager.register_algorithm("super_pixel", SuperPixelAlgorithm)
    AlgorithmManager.register_algorithm("adaptive", AdaptivePixelationAlgorithm)
    AlgorithmManager.register_algorithm("quantization", QuantizationAlgorithm)
    AlgorithmManager.register_algorithm("voronoi", VoronoiAlgorithm)
    AlgorithmManager.register_algorithm("hexagonal", HexagonalAlgorithm)


# Auto-register on import
register_builtin_algorithms()
