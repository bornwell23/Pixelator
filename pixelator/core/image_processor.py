"""
Core image processing module for the PixelArt Creator.

This module provides the main ImageProcessor class that handles:
- Loading and saving images
- Applying pixelation algorithms
- Managing color palettes
- Applying filters and effects
"""

import os
from typing import Optional, Tuple, List, Union
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path

from ..algorithms.base import PixelationAlgorithm
from ..palettes.color_palette import ColorPalette
from ..filters.image_filters import ImageFilters
from ..utils.exceptions import PixelArtError, UnsupportedFormatError


class ImageProcessor:
    """Main image processing class for pixelation and effects."""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    def __init__(self):
        """Initialize the image processor."""
        self.current_image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.processed_image: Optional[Image.Image] = None
        self.algorithm: Optional[PixelationAlgorithm] = None
        self.palette: Optional[ColorPalette] = None
        
    def load_image(self, file_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            PixelArtError: If the image cannot be loaded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(f"Unsupported format: {file_path.suffix}")
            
        try:
            image = Image.open(file_path)
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if image.mode not in ('RGB', 'RGBA'):
                if image.mode == 'P' and 'transparency' in image.info:
                    image = image.convert('RGBA')
                else:
                    image = image.convert('RGB')
                    
            self.original_image = image.copy()
            self.current_image = image.copy()
            return image
            
        except Exception as e:
            raise PixelArtError(f"Failed to load image: {e}")
    
    def save_image(self, output_path: Union[str, Path], 
                   image: Optional[Image.Image] = None,
                   quality: int = 95) -> None:
        """
        Save an image to file.
        
        Args:
            output_path: Path where to save the image
            image: Image to save (uses current_image if None)
            quality: JPEG quality (1-100)
            
        Raises:
            PixelArtError: If no image to save or save fails
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            raise PixelArtError("No image to save")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Handle different formats
            save_kwargs = {}
            if output_path.suffix.lower() in ('.jpg', '.jpeg'):
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
                # Convert RGBA to RGB for JPEG
                if image.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
                    image = background
            elif output_path.suffix.lower() == '.png':
                save_kwargs['optimize'] = True
                
            image.save(output_path, **save_kwargs)
            
        except Exception as e:
            raise PixelArtError(f"Failed to save image: {e}")
    
    def pixelate(self, algorithm: PixelationAlgorithm, 
                 target_size: Optional[Tuple[int, int]] = None,
                 scale_factor: Optional[float] = None,
                 resize_to_target: bool = False) -> Image.Image:
        """
        Apply pixelation to the current image.
        
        Args:
            algorithm: Pixelation algorithm to use
            target_size: Target size (width, height) for pixelation
            scale_factor: Scale factor for pixelation (alternative to target_size)
            resize_to_target: If True, resize final image to target_size instead of original size
            
        Returns:
            Pixelated image
            
        Raises:
            PixelArtError: If no image loaded or parameters invalid
        """
        if self.current_image is None:
            raise PixelArtError("No image loaded")
            
        if target_size is None and scale_factor is None:
            raise PixelArtError("Either target_size or scale_factor must be provided")
            
        if target_size is None:
            # Calculate target size from scale factor
            width, height = self.current_image.size
            target_size = (int(width * scale_factor), int(height * scale_factor))
            
        try:
            self.algorithm = algorithm
            
            if resize_to_target:
                # Apply pixelation and keep the target size
                self.processed_image = algorithm.apply_with_target_size(self.current_image, target_size)
            else:
                # Apply pixelation but preserve original dimensions (default behavior)
                self.processed_image = algorithm.apply(self.current_image, target_size)
                
            self.current_image = self.processed_image.copy()
            return self.processed_image
            
        except Exception as e:
            raise PixelArtError(f"Pixelation failed: {e}")
    
    def apply_palette(self, palette: ColorPalette) -> Image.Image:
        """
        Apply a color palette to the current image.
        
        Args:
            palette: Color palette to apply
            
        Returns:
            Image with applied palette
            
        Raises:
            PixelArtError: If no image loaded
        """
        if self.current_image is None:
            raise PixelArtError("No image loaded")
            
        try:
            self.palette = palette
            self.current_image = palette.apply_to_image(self.current_image)
            return self.current_image
            
        except Exception as e:
            raise PixelArtError(f"Palette application failed: {e}")
    
    def apply_filter(self, filter_name: str, **kwargs) -> Image.Image:
        """
        Apply an image filter to the current image.
        
        Args:
            filter_name: Name of the filter to apply
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered image
            
        Raises:
            PixelArtError: If no image loaded or filter fails
        """
        if self.current_image is None:
            raise PixelArtError("No image loaded")
            
        try:
            self.current_image = ImageFilters.apply_filter(
                self.current_image, filter_name, **kwargs
            )
            return self.current_image
            
        except Exception as e:
            raise PixelArtError(f"Filter application failed: {e}")
    
    def reset_to_original(self) -> Image.Image:
        """
        Reset the current image to the original loaded image.
        
        Returns:
            Original image
            
        Raises:
            PixelArtError: If no original image
        """
        if self.original_image is None:
            raise PixelArtError("No original image to reset to")
            
        self.current_image = self.original_image.copy()
        self.processed_image = None
        return self.current_image
    
    def get_image_info(self) -> dict:
        """
        Get information about the current image.
        
        Returns:
            Dictionary with image information
        """
        if self.current_image is None:
            return {}
            
        return {
            'size': self.current_image.size,
            'mode': self.current_image.mode,
            'format': getattr(self.current_image, 'format', None),
            'has_transparency': self.current_image.mode in ('RGBA', 'LA') or 
                              (self.current_image.mode == 'P' and 'transparency' in self.current_image.info)
        }
    
    def preview_pixelation(self, algorithm: PixelationAlgorithm,
                          target_size: Optional[Tuple[int, int]] = None,
                          scale_factor: Optional[float] = None) -> Image.Image:
        """
        Preview pixelation without modifying the current image.
        
        Args:
            algorithm: Pixelation algorithm to use
            target_size: Target size for pixelation
            scale_factor: Scale factor for pixelation
            
        Returns:
            Previewed pixelated image
        """
        if self.current_image is None:
            raise PixelArtError("No image loaded")
            
        # Work on a copy for preview
        temp_processor = ImageProcessor()
        temp_processor.current_image = self.current_image.copy()
        
        return temp_processor.pixelate(algorithm, target_size, scale_factor)
