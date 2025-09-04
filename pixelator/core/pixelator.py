"""
Main pixelation orchestrator that coordinates different algorithms and processing steps.
"""

from typing import Optional, Dict, Any, Callable
from enum import Enum
import logging
from PIL import Image

from .processor import ImageProcessor
from ..algorithms.base import BaseAlgorithm

logger = logging.getLogger(__name__)


class PixelationMethod(Enum):
    """Available pixelation methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"
    CUSTOM = "custom"


class Pixelator:
    """
    Main orchestrator for the pixelation process.
    
    Coordinates between image processing, algorithms, and palette management
    to create the final pixel art output.
    """
    
    def __init__(self, image_processor: Optional[ImageProcessor] = None):
        """
        Initialize the pixelator.
        
        Args:
            image_processor: ImageProcessor instance to use
        """
        self.image_processor = image_processor or ImageProcessor()
        self.algorithms: Dict[str, BaseAlgorithm] = {}
        self.current_method = PixelationMethod.NEAREST_NEIGHBOR
        self.progress_callback: Optional[Callable[[float], None]] = None
        
        # Default parameters
        self.default_params = {
            'scale_factor': 8,
            'preserve_aspect_ratio': True,
            'apply_palette': False,
            'palette_colors': 16,
            'dithering': False
        }
        
    def register_algorithm(self, name: str, algorithm: BaseAlgorithm) -> None:
        """
        Register a pixelation algorithm.
        
        Args:
            name: Name of the algorithm
            algorithm: Algorithm instance
        """
        self.algorithms[name] = algorithm
        logger.info(f"Registered algorithm: {name}")
    
    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Set callback for progress updates.
        
        Args:
            callback: Function that accepts progress as float (0.0 to 1.0)
        """
        self.progress_callback = callback
    
    def pixelate(self, 
                 method: PixelationMethod = PixelationMethod.NEAREST_NEIGHBOR,
                 **kwargs) -> Optional[Image.Image]:
        """
        Apply pixelation to the current image.
        
        Args:
            method: Pixelation method to use
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Pixelated image or None if failed
        """
        try:
            if self.image_processor.current_image is None:
                logger.error("No image loaded for pixelation")
                return None
                
            # Merge default parameters with provided ones
            params = {**self.default_params, **kwargs}
            
            # Update progress
            self._update_progress(0.1)
            
            # Get algorithm
            algorithm = self._get_algorithm(method)
            if algorithm is None:
                logger.error(f"Algorithm not found: {method.value}")
                return None
                
            # Apply pixelation
            self._update_progress(0.3)
            
            result = algorithm.apply(
                self.image_processor.current_image,
                **params
            )
            
            self._update_progress(0.8)
            
            # Update current image
            if result is not None:
                self.image_processor.current_image = result
                
            self._update_progress(1.0)
            
            logger.info(f"Applied pixelation: {method.value}")
            return result
            
        except Exception as e:
            logger.error(f"Pixelation failed: {e}")
            return None
    
    def batch_pixelate(self, 
                       input_paths: list,
                       output_dir: str,
                       method: PixelationMethod = PixelationMethod.NEAREST_NEIGHBOR,
                       **kwargs) -> Dict[str, bool]:
        """
        Apply pixelation to multiple images.
        
        Args:
            input_paths: List of input image paths
            output_dir: Output directory for processed images
            method: Pixelation method to use
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary mapping input paths to success status
        """
        results = {}
        total_images = len(input_paths)
        
        for i, input_path in enumerate(input_paths):
            try:
                # Load image
                if not self.image_processor.load_image(input_path):
                    results[input_path] = False
                    continue
                    
                # Apply pixelation
                pixelated = self.pixelate(method, **kwargs)
                if pixelated is None:
                    results[input_path] = False
                    continue
                    
                # Generate output path
                from pathlib import Path
                input_file = Path(input_path)
                output_path = Path(output_dir) / f"{input_file.stem}_pixelated{input_file.suffix}"
                
                # Save result
                success = self.image_processor.save_image(output_path, pixelated)
                results[input_path] = success
                
                # Update overall progress
                overall_progress = (i + 1) / total_images
                self._update_progress(overall_progress)
                
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                results[input_path] = False
                
        return results
    
    def preview_pixelation(self,
                          method: PixelationMethod = PixelationMethod.NEAREST_NEIGHBOR,
                          preview_size: tuple = (400, 400),
                          **kwargs) -> Optional[Image.Image]:
        """
        Generate a preview of pixelation without modifying the current image.
        
        Args:
            method: Pixelation method to use
            preview_size: Size for preview image
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Preview image or None if failed
        """
        try:
            if self.image_processor.original_image is None:
                return None
                
            # Create a copy for preview
            original = self.image_processor.current_image
            
            # Resize for preview if image is large
            preview_image = self.image_processor.original_image.copy()
            if max(preview_image.size) > max(preview_size):
                preview_image.thumbnail(preview_size, Image.LANCZOS)
                
            # Temporarily set as current image
            self.image_processor.current_image = preview_image
            
            # Apply pixelation
            result = self.pixelate(method, **kwargs)
            
            # Restore original current image
            self.image_processor.current_image = original
            
            return result
            
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            return None
    
    def get_available_methods(self) -> list:
        """
        Get list of available pixelation methods.
        
        Returns:
            List of method names
        """
        return [method.value for method in PixelationMethod]
    
    def get_method_parameters(self, method: PixelationMethod) -> Dict[str, Any]:
        """
        Get parameters for a specific method.
        
        Args:
            method: Pixelation method
            
        Returns:
            Dictionary of parameter information
        """
        algorithm = self._get_algorithm(method)
        if algorithm is None:
            return {}
            
        return getattr(algorithm, 'get_parameters', lambda: {})()
    
    def _get_algorithm(self, method: PixelationMethod) -> Optional[BaseAlgorithm]:
        """
        Get algorithm instance for the specified method.
        
        Args:
            method: Pixelation method
            
        Returns:
            Algorithm instance or None if not found
        """
        return self.algorithms.get(method.value)
    
    def _update_progress(self, progress: float) -> None:
        """
        Update progress if callback is set.
        
        Args:
            progress: Progress value (0.0 to 1.0)
        """
        if self.progress_callback:
            self.progress_callback(min(1.0, max(0.0, progress)))
