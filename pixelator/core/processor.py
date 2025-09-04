"""
Image processor module for handling image I/O and basic operations.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import logging
from PIL import Image, ImageOps
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image loading, validation, and basic operations.
    
    Supports multiple image formats and provides utilities for
    image manipulation and conversion.
    """
    
    SUPPORTED_FORMATS = {
        'PNG', 'JPEG', 'JPG', 'BMP', 'GIF', 'TIFF', 'TIF', 'WEBP'
    }
    
    def __init__(self):
        """Initialize the image processor."""
        self.current_image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.image_path: Optional[Path] = None
        
    def load_image(self, image_path: Union[str, Path]) -> bool:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image loaded successfully, False otherwise
        """
        try:
            path = Path(image_path)
            
            if not path.exists():
                logger.error(f"Image file not found: {path}")
                return False
                
            if not self._is_supported_format(path):
                logger.error(f"Unsupported image format: {path.suffix}")
                return False
                
            # Load and convert to RGB if necessary
            image = Image.open(path)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
                
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.image_path = path
            
            logger.info(f"Loaded image: {path} ({image.size[0]}x{image.size[1]})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return False
    
    def save_image(self, output_path: Union[str, Path], 
                   image: Optional[Image.Image] = None) -> bool:
        """
        Save the current or provided image to file.
        
        Args:
            output_path: Path where to save the image
            image: Image to save (uses current_image if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            path = Path(output_path)
            save_image = image or self.current_image
            
            if save_image is None:
                logger.error("No image to save")
                return False
                
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to RGB if saving as JPEG
            if path.suffix.upper() in ('.JPG', '.JPEG'):
                if save_image.mode == 'RGBA':
                    # Create white background for transparency
                    rgb_image = Image.new('RGB', save_image.size, (255, 255, 255))
                    rgb_image.paste(save_image, mask=save_image.split()[-1])
                    save_image = rgb_image
                    
            save_image.save(path)
            logger.info(f"Saved image: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image {output_path}: {e}")
            return False
    
    def get_image_info(self) -> dict:
        """
        Get information about the current image.
        
        Returns:
            Dictionary containing image information
        """
        if self.current_image is None:
            return {}
            
        return {
            'size': self.current_image.size,
            'mode': self.current_image.mode,
            'format': getattr(self.current_image, 'format', 'Unknown'),
            'path': str(self.image_path) if self.image_path else None,
            'has_transparency': self.current_image.mode in ('RGBA', 'LA', 'P')
        }
    
    def resize_image(self, size: Tuple[int, int], 
                     resample: int = Image.NEAREST) -> bool:
        """
        Resize the current image.
        
        Args:
            size: New size as (width, height)
            resample: Resampling algorithm
            
        Returns:
            True if resized successfully, False otherwise
        """
        try:
            if self.current_image is None:
                logger.error("No image loaded")
                return False
                
            self.current_image = self.current_image.resize(size, resample)
            return True
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return False
    
    def to_numpy(self, image: Optional[Image.Image] = None) -> np.ndarray:
        """
        Convert image to numpy array.
        
        Args:
            image: Image to convert (uses current_image if None)
            
        Returns:
            Numpy array representation of the image
        """
        use_image = image or self.current_image
        if use_image is None:
            raise ValueError("No image available")
            
        return np.array(use_image)
    
    def from_numpy(self, array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            array: Numpy array to convert
            
        Returns:
            PIL Image object
        """
        return Image.fromarray(array.astype(np.uint8))
    
    def reset_to_original(self) -> bool:
        """
        Reset current image to original.
        
        Returns:
            True if reset successfully, False otherwise
        """
        if self.original_image is None:
            logger.error("No original image available")
            return False
            
        self.current_image = self.original_image.copy()
        return True
    
    def _is_supported_format(self, path: Path) -> bool:
        """
        Check if the file format is supported.
        
        Args:
            path: File path to check
            
        Returns:
            True if format is supported, False otherwise
        """
        return path.suffix.upper().lstrip('.') in self.SUPPORTED_FORMATS
    
    def validate_image_size(self, max_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Validate that the current image size is within limits.
        
        Args:
            max_size: Maximum allowed size as (width, height)
            
        Returns:
            True if valid, False otherwise
        """
        if self.current_image is None:
            return False
            
        if max_size is None:
            max_size = (8192, 8192)  # Default 8K limit
            
        width, height = self.current_image.size
        max_width, max_height = max_size
        
        return width <= max_width and height <= max_height
