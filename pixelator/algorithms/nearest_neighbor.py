"""
Nearest neighbor pixelation algorithm.

This algorithm provides simple downsampling and upsampling using
nearest neighbor interpolation for crisp pixel art effects.
"""

from typing import Dict, Any, Tuple
from PIL import Image
from .base import PixelationAlgorithm


class NearestNeighborAlgorithm(PixelationAlgorithm):
    """
    Simple nearest neighbor pixelation algorithm.
    
    Downsamples the image to a lower resolution and then upsamples
    back to the original size using nearest neighbor interpolation.
    """
    
    def __init__(self):
        """Initialize the nearest neighbor algorithm."""
        super().__init__("Nearest Neighbor")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply nearest neighbor pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size (width, height)
            
        Returns:
            Pixelated image at original size
        """
        original_size = image.size
        
        # Downscale to target size
        downscaled = image.resize(target_size, Image.NEAREST)
        
        # Upscale back to original size
        upscaled = downscaled.resize(original_size, Image.NEAREST)
        
        return upscaled
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        return params
    
    def apply_with_target_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply nearest neighbor pixelation and keep target size.
        
        Args:
            image: Input image
            target_size: Target size for final output
            
        Returns:
            Pixelated image at target size
        """
        # Just downscale to target size with nearest neighbor
        return image.resize(target_size, Image.NEAREST)


class BilinearPixelationAlgorithm(PixelationAlgorithm):
    """
    Bilinear pixelation algorithm.
    
    Uses bilinear interpolation for smoother downsampling,
    then nearest neighbor for upsampling to maintain pixel edges.
    """
    
    def __init__(self):
        """Initialize the bilinear algorithm."""
        super().__init__("Bilinear")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply bilinear pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size (width, height)
            
        Returns:
            Pixelated image at original size
        """
        original_size = image.size
        
        # Downscale with bilinear interpolation for smoothing
        downscaled = image.resize(target_size, Image.BILINEAR)
        
        # Upscale with nearest neighbor to maintain pixel edges
        upscaled = downscaled.resize(original_size, Image.NEAREST)
        
        return upscaled
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        return params
    
    def apply_with_target_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply bilinear pixelation and keep target size.
        
        Args:
            image: Input image
            target_size: Target size for final output
            
        Returns:
            Pixelated image at target size
        """
        # Downscale with bilinear interpolation
        return image.resize(target_size, Image.BILINEAR)


class LanczosPixelationAlgorithm(PixelationAlgorithm):
    """
    Lanczos pixelation algorithm.
    
    Uses Lanczos resampling for high-quality downsampling,
    then nearest neighbor for upsampling.
    """
    
    def __init__(self):
        """Initialize the Lanczos algorithm."""
        super().__init__("Lanczos")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply Lanczos pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size (width, height)
            
        Returns:
            Pixelated image at original size
        """
        original_size = image.size
        
        # Downscale with Lanczos for high quality
        downscaled = image.resize(target_size, Image.LANCZOS)
        
        # Upscale with nearest neighbor to maintain pixel edges
        upscaled = downscaled.resize(original_size, Image.NEAREST)
        
        return upscaled
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        return params
    
    def apply_with_target_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply Lanczos pixelation and keep target size.
        
        Args:
            image: Input image
            target_size: Target size for final output
            
        Returns:
            Pixelated image at target size
        """
        # Downscale with Lanczos for high quality
        return image.resize(target_size, Image.LANCZOS)
