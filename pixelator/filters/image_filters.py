"""
Image filters and effects for pixel art creation.

This module provides various filters and effects that can be applied
to images to enhance the pixel art aesthetic.
"""

from typing import Dict, Any, Optional, Callable
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from ..utils.exceptions import PixelArtError


class ImageFilters:
    """Collection of image filters and effects for pixel art."""
    
    # Registry of available filters
    _filters: Dict[str, Callable] = {}
    
    @classmethod
    def register_filter(cls, name: str):
        """Decorator to register a filter function."""
        def decorator(func):
            cls._filters[name] = func
            return func
        return decorator
    
    @classmethod
    def apply_filter(cls, image: Image.Image, filter_name: str, **kwargs) -> Image.Image:
        """
        Apply a filter to an image.
        
        Args:
            image: Input image
            filter_name: Name of the filter to apply
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered image
            
        Raises:
            PixelArtError: If filter not found or application fails
        """
        if filter_name not in cls._filters:
            available = ', '.join(cls._filters.keys())
            raise PixelArtError(f"Filter '{filter_name}' not found. Available: {available}")
        
        try:
            return cls._filters[filter_name](image, **kwargs)
        except Exception as e:
            raise PixelArtError(f"Failed to apply filter '{filter_name}': {e}")
    
    @classmethod
    def get_available_filters(cls) -> Dict[str, str]:
        """
        Get list of available filters with descriptions.
        
        Returns:
            Dictionary mapping filter names to descriptions
        """
        descriptions = {
            'blur': 'Apply Gaussian blur to soften edges',
            'sharpen': 'Enhance image sharpness and edge definition',
            'edge_enhance': 'Enhance edges while preserving detail',
            'emboss': 'Create embossed 3D effect',
            'find_edges': 'Detect and highlight edges',
            'contrast': 'Adjust image contrast',
            'brightness': 'Adjust image brightness',
            'saturation': 'Adjust color saturation',
            'posterize': 'Reduce number of color levels',
            'solarize': 'Apply solarization effect',
            'invert': 'Invert image colors',
            'grayscale': 'Convert to grayscale',
            'sepia': 'Apply sepia tone effect',
            'noise': 'Add random noise',
            'vignette': 'Add dark vignette around edges',
            'scan_lines': 'Add CRT-style scan lines',
            'chromatic_aberration': 'Simulate lens chromatic aberration',
            'pixelate_mosaic': 'Create mosaic-style pixelation',
            'halftone': 'Apply halftone dot pattern effect',
            'cross_hatch': 'Apply cross-hatching pen and ink effect',
            'oil_painting': 'Apply oil painting artistic effect',
            'film_grain': 'Add film grain texture',
            'color_shift': 'Shift colors in HSV color space',
            'texture_overlay': 'Overlay texture pattern (paper/canvas/fabric)'
        }
        return {name: descriptions.get(name, 'No description') for name in cls._filters.keys()}


# Register built-in filters
@ImageFilters.register_filter('blur')
def blur_filter(image: Image.Image, radius: float = 1.0) -> Image.Image:
    """Apply Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


@ImageFilters.register_filter('sharpen')
def sharpen_filter(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Apply sharpening filter."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(1.0 + factor)


@ImageFilters.register_filter('edge_enhance')
def edge_enhance_filter(image: Image.Image) -> Image.Image:
    """Enhance edges."""
    return image.filter(ImageFilter.EDGE_ENHANCE)


@ImageFilters.register_filter('emboss')
def emboss_filter(image: Image.Image) -> Image.Image:
    """Apply emboss effect."""
    return image.filter(ImageFilter.EMBOSS)


@ImageFilters.register_filter('find_edges')
def find_edges_filter(image: Image.Image) -> Image.Image:
    """Find and highlight edges."""
    return image.filter(ImageFilter.FIND_EDGES)


@ImageFilters.register_filter('contrast')
def contrast_filter(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust contrast. Factor > 1 increases contrast, < 1 decreases."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


@ImageFilters.register_filter('brightness')
def brightness_filter(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust brightness. Factor > 1 brightens, < 1 darkens."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


@ImageFilters.register_filter('saturation')
def saturation_filter(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust saturation. Factor > 1 increases saturation, < 1 decreases."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


@ImageFilters.register_filter('posterize')
def posterize_filter(image: Image.Image, bits: int = 4) -> Image.Image:
    """Reduce number of color levels."""
    return ImageOps.posterize(image, bits)


@ImageFilters.register_filter('solarize')
def solarize_filter(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Apply solarization effect."""
    return ImageOps.solarize(image, threshold)


@ImageFilters.register_filter('invert')
def invert_filter(image: Image.Image) -> Image.Image:
    """Invert image colors."""
    return ImageOps.invert(image)


@ImageFilters.register_filter('grayscale')
def grayscale_filter(image: Image.Image) -> Image.Image:
    """Convert to grayscale."""
    return ImageOps.grayscale(image).convert('RGB')


@ImageFilters.register_filter('sepia')
def sepia_filter(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Apply sepia tone effect."""
    # Convert to grayscale first
    gray = ImageOps.grayscale(image)
    
    # Create sepia-toned image
    sepia = Image.new('RGB', gray.size)
    sepia_pixels = []
    
    for pixel in gray.getdata():
        # Sepia tone formula
        r = min(255, int(pixel * (1.0 + 0.3 * intensity)))
        g = min(255, int(pixel * (1.0 + 0.1 * intensity)))
        b = min(255, int(pixel * (1.0 - 0.2 * intensity)))
        sepia_pixels.append((r, g, b))
    
    sepia.putdata(sepia_pixels)
    return sepia


@ImageFilters.register_filter('noise')
def noise_filter(image: Image.Image, intensity: float = 0.1) -> Image.Image:
    """Add random noise to image."""
    img_array = np.array(image)
    noise = np.random.randint(-int(255 * intensity), int(255 * intensity) + 1, 
                              img_array.shape, dtype=np.int16)
    
    noisy = img_array.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy)


@ImageFilters.register_filter('vignette')
def vignette_filter(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Add dark vignette around edges."""
    width, height = image.size
    img_array = np.array(image).astype(np.float64)
    
    # Create vignette mask
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize distance and create vignette
    vignette = 1 - (distance / max_distance) * intensity
    vignette = np.clip(vignette, 0, 1)
    
    # Apply vignette to all channels
    for channel in range(img_array.shape[2]):
        img_array[:, :, channel] *= vignette
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


@ImageFilters.register_filter('scan_lines')
def scan_lines_filter(image: Image.Image, line_height: int = 2, 
                     opacity: float = 0.3) -> Image.Image:
    """Add CRT-style scan lines."""
    width, height = image.size
    img_array = np.array(image).astype(np.float64)
    
    # Create scan line pattern
    for y in range(0, height, line_height * 2):
        if y + line_height < height:
            img_array[y:y+line_height, :] *= (1 - opacity)
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


@ImageFilters.register_filter('chromatic_aberration')
def chromatic_aberration_filter(image: Image.Image, offset: int = 2) -> Image.Image:
    """Simulate chromatic aberration effect."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Split into RGB channels
    r, g, b = image.split()
    
    # Shift channels slightly
    # Red channel - shift right
    r_shifted = Image.new('L', r.size, 0)
    r_shifted.paste(r, (offset, 0))
    
    # Blue channel - shift left
    b_shifted = Image.new('L', b.size, 0)
    b_shifted.paste(b, (-offset, 0))
    
    # Recombine channels
    return Image.merge('RGB', (r_shifted, g, b_shifted))


@ImageFilters.register_filter('pixelate_mosaic')
def pixelate_mosaic_filter(image: Image.Image, block_size: int = 8) -> Image.Image:
    """Create mosaic-style pixelation effect."""
    width, height = image.size
    img_array = np.array(image)
    
    # Process in blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Get block boundaries
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            # Get block and calculate average color
            block = img_array[y:y_end, x:x_end]
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            
            # Fill block with average color
            img_array[y:y_end, x:x_end] = avg_color
    
    return Image.fromarray(img_array)


@ImageFilters.register_filter('halftone')
def halftone_filter(image: Image.Image, dot_size: int = 4) -> Image.Image:
    """Apply halftone dot pattern effect."""
    # Ensure dot_size is an integer
    dot_size = int(dot_size)
    
    # Convert to grayscale for halftone calculation
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    gray = image.convert('L')
    img_array = np.array(image)
    gray_array = np.array(gray)
    height, width = gray_array.shape
    
    result = np.zeros_like(img_array)
    
    # Create halftone pattern
    for y in range(0, height, dot_size):
        for x in range(0, width, dot_size):
            # Get block
            y_end = min(y + dot_size, height)
            x_end = min(x + dot_size, width)
            
            # Calculate average brightness
            block = gray_array[y:y_end, x:x_end]
            avg_brightness = np.mean(block) / 255.0
            
            # Create circular dot based on brightness
            dot_radius = int((dot_size // 2) * avg_brightness)
            center_y, center_x = int((y + y_end) // 2), int((x + x_end) // 2)
            
            # Get original color
            color_block = img_array[y:y_end, x:x_end]
            avg_color = np.mean(color_block, axis=(0, 1))
            
            # Draw dot
            for dy in range(y, y_end):
                for dx in range(x, x_end):
                    distance = np.sqrt((dy - center_y)**2 + (dx - center_x)**2)
                    if distance <= dot_radius:
                        result[dy, dx] = avg_color
                    else:
                        result[dy, dx] = [255, 255, 255]  # White background
    
    return Image.fromarray(result.astype(np.uint8))


@ImageFilters.register_filter('cross_hatch')
def cross_hatch_filter(image: Image.Image, line_density: int = 8) -> Image.Image:
    """Apply cross-hatching pen and ink effect."""
    # Ensure line_density is an integer
    line_density = int(line_density)
    
    # Convert to grayscale
    gray = image.convert('L')
    gray_array = np.array(gray)
    height, width = gray_array.shape
    
    result = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Create cross-hatch pattern based on brightness
    for y in range(height):
        for x in range(width):
            brightness = gray_array[y, x] / 255.0
            
            # Determine line density based on brightness
            if brightness < 0.2:
                # Very dark - dense cross-hatching
                if (x % line_density < 2) or (y % line_density < 2) or \
                   ((x + y) % line_density < 2) or ((x - y) % line_density < 2):
                    result[y, x] = [0, 0, 0]
            elif brightness < 0.4:
                # Dark - medium cross-hatching
                if (x % line_density < 1) or (y % line_density < 1) or \
                   ((x + y) % (line_density * 2) < 1):
                    result[y, x] = [0, 0, 0]
            elif brightness < 0.6:
                # Medium - light hatching
                if (x % (line_density * 2) < 1) or (y % (line_density * 2) < 1):
                    result[y, x] = [0, 0, 0]
            elif brightness < 0.8:
                # Light - very light hatching
                if x % (line_density * 4) < 1:
                    result[y, x] = [128, 128, 128]
    
    return Image.fromarray(result)


@ImageFilters.register_filter('oil_painting')
def oil_painting_filter(image: Image.Image, radius: int = 4, intensity: int = 20) -> Image.Image:
    """Apply oil painting effect."""
    # Ensure parameters are integers
    radius = int(radius)
    intensity = int(intensity)
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    result = np.zeros_like(img_array)
    
    for y in range(height):
        for x in range(width):
            # Define search area
            y_start = max(0, y - radius)
            y_end = min(height, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(width, x + radius + 1)
            
            # Extract neighborhood
            neighborhood = img_array[y_start:y_end, x_start:x_end]
            
            # Quantize colors and find most frequent
            if len(img_array.shape) == 3:
                # Quantize each channel
                quantized = (neighborhood // intensity) * intensity
                
                # Find most common color
                colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
                most_common_color = colors[np.argmax(counts)]
                result[y, x] = most_common_color
            else:
                # Grayscale
                quantized = (neighborhood // intensity) * intensity
                values, counts = np.unique(quantized, return_counts=True)
                result[y, x] = values[np.argmax(counts)]
    
    return Image.fromarray(result.astype(np.uint8))


@ImageFilters.register_filter('film_grain')
def film_grain_filter(image: Image.Image, intensity: float = 0.1, grain_size: int = 1) -> Image.Image:
    """Add film grain texture."""
    # Ensure grain_size is an integer
    grain_size = int(grain_size)
    
    img_array = np.array(image).astype(np.float64)
    
    # Generate grain pattern
    if grain_size == 1:
        grain = np.random.normal(0, intensity * 255, img_array.shape)
    else:
        # Create larger grain pattern
        small_grain = np.random.normal(0, intensity * 255, 
                                     (img_array.shape[0]//grain_size, 
                                      img_array.shape[1]//grain_size,
                                      img_array.shape[2] if len(img_array.shape) == 3 else 1))
        # Resize to match image
        from PIL import Image as PILImage
        if len(img_array.shape) == 3:
            grain = np.array(PILImage.fromarray(small_grain.astype(np.uint8)).resize(
                (img_array.shape[1], img_array.shape[0]), PILImage.NEAREST))
        else:
            grain = np.array(PILImage.fromarray(small_grain.squeeze().astype(np.uint8)).resize(
                (img_array.shape[1], img_array.shape[0]), PILImage.NEAREST))
            if len(img_array.shape) == 2:
                grain = grain[:, :, np.newaxis]
    
    # Apply grain
    result = img_array + grain
    result = np.clip(result, 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))


@ImageFilters.register_filter('color_shift')
def color_shift_filter(image: Image.Image, hue_shift: float = 0.0, 
                      saturation_factor: float = 1.0, value_factor: float = 1.0) -> Image.Image:
    """Shift colors in HSV color space."""
    import colorsys
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    result = np.zeros_like(img_array)
    
    # Convert to HSV and apply shifts
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            r, g, b = img_array[y, x] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # Apply shifts
            h = (h + hue_shift) % 1.0
            s = max(0, min(1, s * saturation_factor))
            v = max(0, min(1, v * value_factor))
            
            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            result[y, x] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return Image.fromarray(result.astype(np.uint8))


@ImageFilters.register_filter('texture_overlay')
def texture_overlay_filter(image: Image.Image, texture_type: str = 'paper', 
                          intensity: float = 0.3) -> Image.Image:
    """Overlay texture pattern."""
    # Handle numeric texture types from CLI
    if isinstance(texture_type, (int, float)):
        texture_types = ['paper', 'canvas', 'fabric']
        texture_type = texture_types[int(texture_type) % len(texture_types)]
    
    img_array = np.array(image).astype(np.float64)
    height, width = img_array.shape[:2]
    
    # Generate texture based on type
    if texture_type == 'paper':
        # Paper texture using Perlin-like noise
        texture = np.random.random((height, width)) * 0.3 + 0.7
        # Add some structure
        for i in range(0, height, 8):
            for j in range(0, width, 12):
                texture[i:i+2, j:j+2] *= 0.9
    elif texture_type == 'canvas':
        # Canvas weave pattern
        texture = np.ones((height, width))
        for i in range(0, height, 4):
            texture[i:i+1, :] *= 0.9
        for j in range(0, width, 4):
            texture[:, j:j+1] *= 0.9
    else:  # fabric
        # Fabric texture
        texture = np.random.random((height, width)) * 0.2 + 0.9
        # Add diagonal weave
        for i in range(height):
            for j in range(width):
                if (i + j) % 6 < 2:
                    texture[i, j] *= 0.85
    
    # Apply texture
    if len(img_array.shape) == 3:
        for channel in range(3):
            img_array[:, :, channel] = img_array[:, :, channel] * (
                1 - intensity + intensity * texture)
    else:
        img_array = img_array * (1 - intensity + intensity * texture)
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class FilterChain:
    """Chain multiple filters together for complex effects."""
    
    def __init__(self):
        """Initialize empty filter chain."""
        self.filters = []
    
    def add_filter(self, filter_name: str, **kwargs) -> 'FilterChain':
        """
        Add a filter to the chain.
        
        Args:
            filter_name: Name of the filter
            **kwargs: Filter parameters
            
        Returns:
            Self for method chaining
        """
        self.filters.append((filter_name, kwargs))
        return self
    
    def apply(self, image: Image.Image) -> Image.Image:
        """
        Apply all filters in the chain to an image.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        result = image.copy()
        
        for filter_name, kwargs in self.filters:
            result = ImageFilters.apply_filter(result, filter_name, **kwargs)
        
        return result
    
    def clear(self) -> None:
        """Clear all filters from the chain."""
        self.filters.clear()
    
    def remove_filter(self, index: int) -> None:
        """
        Remove filter at specified index.
        
        Args:
            index: Index of filter to remove
        """
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
    
    def get_filters(self) -> list:
        """Get list of filters in the chain."""
        return self.filters.copy()
    
    def __len__(self) -> int:
        """Get number of filters in chain."""
        return len(self.filters)
    
    def __str__(self) -> str:
        """String representation of filter chain."""
        if not self.filters:
            return "Empty filter chain"
        
        filter_names = [f[0] for f in self.filters]
        return f"Filter chain: {' -> '.join(filter_names)}"
