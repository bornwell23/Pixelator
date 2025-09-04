"""
Base class for all pixelation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """
    Abstract base class for all pixelation algorithms.
    
    Provides a common interface and utilities that all
    pixelation algorithms must implement.
    """
    
    def __init__(self, name: str):
        """
        Initialize the algorithm.
        
        Args:
            name: Name of the algorithm
        """
        self.name = name
        self.parameters = {}
        
    @abstractmethod
    def apply(self, image: Image.Image, **kwargs) -> Optional[Image.Image]:
        """
        Apply the pixelation algorithm to an image.
        
        Args:
            image: Input image to pixelate
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Pixelated image or None if failed
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get algorithm-specific parameters and their descriptions.
        
        Returns:
            Dictionary with parameter information
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and process input parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Validated parameters dictionary
        """
        validated = {}
        param_info = self.get_parameters()
        
        for param_name, param_config in param_info.items():
            value = kwargs.get(param_name, param_config.get('default'))
            
            # Type validation
            expected_type = param_config.get('type', type(value))
            if value is not None and not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid type for {param_name}, using default")
                    value = param_config.get('default')
            
            # Range validation
            min_val = param_config.get('min')
            max_val = param_config.get('max')
            
            if min_val is not None and value is not None and value < min_val:
                logger.warning(f"{param_name} below minimum, using {min_val}")
                value = min_val
                
            if max_val is not None and value is not None and value > max_val:
                logger.warning(f"{param_name} above maximum, using {max_val}")
                value = max_val
            
            validated[param_name] = value
            
        return validated
    
    def calculate_output_size(self, input_size: tuple, scale_factor: int,
                            preserve_aspect_ratio: bool = True) -> tuple:
        """
        Calculate output image size based on scale factor.
        
        Args:
            input_size: Original image size as (width, height)
            scale_factor: Scaling factor for pixelation
            preserve_aspect_ratio: Whether to preserve aspect ratio
            
        Returns:
            Output size as (width, height)
        """
        width, height = input_size
        
        if preserve_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            new_width = max(1, width // scale_factor)
            new_height = max(1, height // scale_factor)
        else:
            # Use exact scaling
            new_width = max(1, width // scale_factor)
            new_height = max(1, height // scale_factor)
            
        return (new_width, new_height)
    
    def log_operation(self, operation: str, **details) -> None:
        """
        Log algorithm operation with details.
        
        Args:
            operation: Operation description
            **details: Additional operation details
        """
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        logger.info(f"{self.name}: {operation} ({detail_str})")


class PixelationAlgorithm(BaseAlgorithm):
    """
    Base class specifically for pixelation algorithms.
    
    Provides common pixelation utilities and standard parameter handling.
    """
    
    def __init__(self, name: str):
        """Initialize pixelation algorithm."""
        super().__init__(name)
        
    def get_common_parameters(self) -> Dict[str, Any]:
        """
        Get common parameters shared by most pixelation algorithms.
        
        Returns:
            Dictionary of common parameter configurations
        """
        return {
            'scale_factor': {
                'type': int,
                'default': 8,
                'min': 1,
                'max': 100,
                'description': 'Factor by which to reduce image resolution'
            },
            'preserve_aspect_ratio': {
                'type': bool,
                'default': True,
                'description': 'Whether to maintain original aspect ratio'
            },
            'resample_method': {
                'type': str,
                'default': 'nearest',
                'choices': ['nearest', 'bilinear', 'bicubic', 'lanczos'],
                'description': 'Resampling method for scaling'
            }
        }
    
    def downscale_image(self, image: Image.Image, target_size: tuple,
                       resample: int = Image.NEAREST) -> Image.Image:
        """
        Downscale image to target size.
        
        Args:
            image: Image to downscale
            target_size: Target size as (width, height)
            resample: Resampling method
            
        Returns:
            Downscaled image
        """
        return image.resize(target_size, resample)
    
    def upscale_image(self, image: Image.Image, target_size: tuple,
                     resample: int = Image.NEAREST) -> Image.Image:
        """
        Upscale image to target size.
        
        Args:
            image: Image to upscale
            target_size: Target size as (width, height)
            resample: Resampling method
            
        Returns:
            Upscaled image
        """
        return image.resize(target_size, resample)
    
    def get_resample_method(self, method_name: str) -> int:
        """
        Convert method name to PIL constant.
        
        Args:
            method_name: Name of resampling method
            
        Returns:
            PIL resampling constant
        """
        methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        return methods.get(method_name.lower(), Image.NEAREST)
    
    def apply_with_target_size(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """
        Apply pixelation and resize to target size (don't upscale back to original).
        
        Args:
            image: Input image
            target_size: Target size for final output
            
        Returns:
            Pixelated image at target size
        """
        # Default implementation: 
        # 1. Apply the normal pixelation algorithm to get the pixelated effect
        # 2. Then resize to target size
        pixelated = self.apply(image, target_size)
        # If the pixelated image is already at target size, return it
        if pixelated.size == target_size:
            return pixelated
        # Otherwise resize to target size
        return pixelated.resize(target_size, Image.NEAREST)
