"""
Advanced pixelation algorithms with edge preservation and color quantization.

This module provides more sophisticated pixelation techniques that
preserve important visual features while creating pixel art effects.
"""

from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import segmentation, filters
from .base import PixelationAlgorithm


class EdgePreservingAlgorithm(PixelationAlgorithm):
    """
    Edge-preserving pixelation algorithm.
    
    Uses edge detection to preserve important boundaries
    while pixelating less important areas.
    """
    
    def __init__(self):
        """Initialize the edge-preserving algorithm."""
        super().__init__("Edge Preserving")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply edge-preserving pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size (width, height)
            
        Returns:
            Pixelated image with preserved edges
        """
        original_size = image.size
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Detect edges
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        edges = filters.sobel(gray)
        
        # Create edge mask (stronger edges = less pixelation)
        edge_threshold = np.percentile(edges, 75)  # Top 25% of edges
        edge_mask = edges > edge_threshold
        
        # Apply different levels of pixelation
        # High pixelation for non-edge areas
        heavily_pixelated = self._simple_pixelate(image, target_size)
        
        # Light pixelation for edge areas
        light_target = (target_size[0] * 2, target_size[1] * 2)
        lightly_pixelated = self._simple_pixelate(image, light_target)
        
        # Blend based on edge mask
        result = self._blend_images(heavily_pixelated, lightly_pixelated, edge_mask)
        
        return result
    
    def _simple_pixelate(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Apply simple pixelation."""
        original_size = image.size
        downscaled = image.resize(target_size, Image.NEAREST)
        return downscaled.resize(original_size, Image.NEAREST)
    
    def _blend_images(self, img1: Image.Image, img2: Image.Image, 
                     mask: np.ndarray) -> Image.Image:
        """Blend two images based on a mask."""
        arr1 = np.array(img1).astype(np.float64)
        arr2 = np.array(img2).astype(np.float64)
        
        # Expand mask to match image channels
        if len(arr1.shape) == 3:
            mask = np.stack([mask] * arr1.shape[2], axis=2)
        
        # Blend images
        result = arr1 * (1 - mask) + arr2 * mask
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'edge_sensitivity': {
                'type': float,
                'default': 0.75,
                'min': 0.1,
                'max': 0.9,
                'description': 'Sensitivity to edge detection (higher = more edge preservation)'
            }
        })
        return params


class SuperPixelAlgorithm(PixelationAlgorithm):
    """
    Super-pixel based pixelation algorithm.
    
    Groups similar pixels into super-pixels and then
    fills each super-pixel with its average color.
    """
    
    def __init__(self):
        """Initialize the super-pixel algorithm."""
        super().__init__("Super Pixel")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply super-pixel pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size (affects number of super-pixels)
            
        Returns:
            Super-pixel pixelated image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate number of super-pixels based on target size
        original_pixels = image.size[0] * image.size[1]
        target_pixels = target_size[0] * target_size[1]
        n_segments = max(50, int(target_pixels / 4))  # Approximate super-pixel count
        
        # Apply SLIC super-pixel segmentation
        segments = segmentation.slic(img_array, n_segments=n_segments, 
                                   compactness=10, sigma=1, start_label=1)
        
        # Fill each segment with its average color
        result = np.zeros_like(img_array)
        
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            if len(img_array.shape) == 3:
                # Color image
                for channel in range(img_array.shape[2]):
                    avg_color = np.mean(img_array[mask, channel])
                    result[mask, channel] = avg_color
            else:
                # Grayscale
                avg_color = np.mean(img_array[mask])
                result[mask] = avg_color
        
        return Image.fromarray(result.astype(np.uint8))
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'compactness': {
                'type': float,
                'default': 10.0,
                'min': 1.0,
                'max': 50.0,
                'description': 'Compactness of super-pixels (higher = more regular shapes)'
            },
            'sigma': {
                'type': float,
                'default': 1.0,
                'min': 0.1,
                'max': 5.0,
                'description': 'Gaussian smoothing parameter'
            }
        })
        return params


class AdaptivePixelationAlgorithm(PixelationAlgorithm):
    """
    Adaptive pixelation algorithm.
    
    Applies different levels of pixelation based on local
    image complexity and detail.
    """
    
    def __init__(self):
        """Initialize the adaptive algorithm."""
        super().__init__("Adaptive")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply adaptive pixelation.
        
        Args:
            image: Input image
            target_size: Base target size for pixelation
            
        Returns:
            Adaptively pixelated image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate local variance to measure complexity
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Calculate local variance using a sliding window
        variance_map = self._calculate_local_variance(gray, window_size=16)
        
        # Normalize variance map
        variance_normalized = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
        
        # Create adaptive pixelation levels
        # High variance (complex areas) = less pixelation
        # Low variance (smooth areas) = more pixelation
        
        # Define multiple pixelation levels
        levels = [
            (target_size, 0.8),  # Heavy pixelation for smooth areas
            ((target_size[0] * 2, target_size[1] * 2), 0.6),  # Medium pixelation
            ((target_size[0] * 4, target_size[1] * 4), 0.4),  # Light pixelation
            (image.size, 0.0)  # No pixelation for very complex areas
        ]
        
        # Apply different pixelation levels
        result_array = np.zeros_like(img_array, dtype=np.float64)
        weight_sum = np.zeros(gray.shape, dtype=np.float64)
        
        for level_size, threshold in levels:
            # Create mask for this level
            mask = (variance_normalized >= threshold) & (variance_normalized < threshold + 0.2)
            
            if np.any(mask):
                # Apply pixelation at this level
                if level_size == image.size:
                    level_result = img_array
                else:
                    level_result = np.array(self._simple_pixelate(image, level_size))
                
                # Accumulate weighted results
                if len(img_array.shape) == 3:
                    for channel in range(img_array.shape[2]):
                        result_array[:, :, channel] += level_result[:, :, channel] * mask
                        weight_sum += mask
                else:
                    result_array += level_result * mask
                    weight_sum += mask
        
        # Normalize by weights
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        if len(img_array.shape) == 3:
            for channel in range(img_array.shape[2]):
                result_array[:, :, channel] /= weight_sum
        else:
            result_array /= weight_sum
        
        return Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    def _calculate_local_variance(self, image: np.ndarray, window_size: int = 16) -> np.ndarray:
        """Calculate local variance using sliding window."""
        # Pad image for border handling
        padded = np.pad(image, window_size//2, mode='reflect')
        
        variance_map = np.zeros_like(image, dtype=np.float64)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extract window
                window = padded[i:i+window_size, j:j+window_size]
                variance_map[i, j] = np.var(window)
        
        return variance_map
    
    def _simple_pixelate(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Apply simple pixelation."""
        original_size = image.size
        downscaled = image.resize(target_size, Image.NEAREST)
        return downscaled.resize(original_size, Image.NEAREST)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'complexity_threshold': {
                'type': float,
                'default': 0.5,
                'min': 0.1,
                'max': 0.9,
                'description': 'Threshold for complexity-based adaptation'
            },
            'window_size': {
                'type': int,
                'default': 16,
                'min': 4,
                'max': 64,
                'description': 'Window size for local complexity analysis'
            }
        })
        return params


class QuantizationAlgorithm(PixelationAlgorithm):
    """
    K-means quantization-based pixelation algorithm.
    
    Reduces colors using K-means clustering before pixelating,
    creating a more coherent color palette.
    """
    
    def __init__(self):
        """Initialize the quantization algorithm."""
        super().__init__("Quantization")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply quantization-based pixelation.
        
        Args:
            image: Input image
            target_size: Target pixelated size
            
        Returns:
            Quantized and pixelated image
        """
        from sklearn.cluster import KMeans
        
        # Convert to numpy array
        img_array = np.array(image)
        original_shape = img_array.shape
        
        # Reshape for clustering
        pixels = img_array.reshape(-1, 3 if len(original_shape) == 3 else 1)
        
        # Determine number of colors based on target size
        target_pixels = target_size[0] * target_size[1]
        n_colors = max(8, min(64, target_pixels // 16))
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Replace pixels with cluster centers
        quantized_pixels = kmeans.cluster_centers_[labels]
        quantized_array = quantized_pixels.reshape(original_shape).astype(np.uint8)
        
        # Convert back to PIL and apply basic pixelation
        quantized_image = Image.fromarray(quantized_array)
        
        # Apply pixelation
        original_size = image.size
        downscaled = quantized_image.resize(target_size, Image.NEAREST)
        return downscaled.resize(original_size, Image.NEAREST)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'n_colors': {
                'type': int,
                'default': 32,
                'min': 4,
                'max': 128,
                'description': 'Number of colors for quantization'
            }
        })
        return params


class VoronoiAlgorithm(PixelationAlgorithm):
    """
    Voronoi diagram-based pixelation algorithm.
    
    Creates irregular pixel shapes based on Voronoi diagrams,
    giving a more organic pixelated look.
    """
    
    def __init__(self):
        """Initialize the Voronoi algorithm."""
        super().__init__("Voronoi")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply Voronoi-based pixelation.
        
        Args:
            image: Input image
            target_size: Affects density of Voronoi cells
            
        Returns:
            Voronoi pixelated image
        """
        from scipy.spatial.distance import cdist
        
        # Convert to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Calculate number of seed points based on target size
        target_pixels = target_size[0] * target_size[1]
        n_points = max(50, min(1000, target_pixels))
        
        # Generate random seed points
        np.random.seed(42)  # For reproducible results
        points = np.random.rand(n_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        pixel_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Find closest seed point for each pixel
        distances = cdist(pixel_coords, points)
        closest_points = np.argmin(distances, axis=1)
        
        # Create result image
        result = np.zeros_like(img_array)
        
        # Fill each Voronoi cell with average color
        for point_idx in range(n_points):
            mask = closest_points == point_idx
            if np.any(mask):
                # Get pixel coordinates for this cell
                cell_pixels = pixel_coords[mask]
                
                if len(img_array.shape) == 3:
                    # Color image
                    for channel in range(img_array.shape[2]):
                        cell_colors = img_array[cell_pixels[:, 1], cell_pixels[:, 0], channel]
                        avg_color = np.mean(cell_colors)
                        result[cell_pixels[:, 1], cell_pixels[:, 0], channel] = avg_color
                else:
                    # Grayscale
                    cell_colors = img_array[cell_pixels[:, 1], cell_pixels[:, 0]]
                    avg_color = np.mean(cell_colors)
                    result[cell_pixels[:, 1], cell_pixels[:, 0]] = avg_color
        
        return Image.fromarray(result.astype(np.uint8))
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'cell_density': {
                'type': float,
                'default': 1.0,
                'min': 0.1,
                'max': 3.0,
                'description': 'Density of Voronoi cells (higher = more cells)'
            }
        })
        return params


class HexagonalAlgorithm(PixelationAlgorithm):
    """
    Hexagonal pixelation algorithm.
    
    Creates hexagonal pixel patterns instead of square pixels,
    giving a unique honeycomb-like appearance.
    """
    
    def __init__(self):
        """Initialize the hexagonal algorithm."""
        super().__init__("Hexagonal")
        
    def apply(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Apply hexagonal pixelation.
        
        Args:
            image: Input image
            target_size: Affects size of hexagonal cells
            
        Returns:
            Hexagonally pixelated image
        """
        # Convert to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Calculate hexagon size based on target size
        scale_factor = max(image.size) / max(target_size)
        hex_size = max(4, int(scale_factor))
        
        # Create result image
        result = np.zeros_like(img_array)
        
        # Hexagonal grid parameters
        hex_width = hex_size * 2
        hex_height = int(hex_size * np.sqrt(3))
        
        # Process in hexagonal pattern
        for row in range(0, height, hex_height // 2):
            for col in range(0, width, hex_width):
                # Offset every other row for hexagonal tiling
                offset = (hex_width // 2) if (row // (hex_height // 2)) % 2 == 1 else 0
                hex_col = col + offset
                
                # Get hexagon bounds
                y_start = max(0, row)
                y_end = min(height, row + hex_height)
                x_start = max(0, hex_col)
                x_end = min(width, hex_col + hex_width)
                
                if y_end > y_start and x_end > x_start:
                    # Create hexagonal mask
                    mask = self._create_hexagon_mask(
                        x_end - x_start, y_end - y_start, hex_size
                    )
                    
                    # Extract region
                    region = img_array[y_start:y_end, x_start:x_end]
                    
                    # Calculate average color within hexagon
                    if len(img_array.shape) == 3:
                        for channel in range(img_array.shape[2]):
                            channel_data = region[:, :, channel]
                            if mask.shape == channel_data.shape:
                                masked_data = channel_data[mask > 0]
                                if len(masked_data) > 0:
                                    avg_color = np.mean(masked_data)
                                    result[y_start:y_end, x_start:x_end, channel][mask > 0] = avg_color
                    else:
                        if mask.shape == region.shape:
                            masked_data = region[mask > 0]
                            if len(masked_data) > 0:
                                avg_color = np.mean(masked_data)
                                result[y_start:y_end, x_start:x_end][mask > 0] = avg_color
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _create_hexagon_mask(self, width: int, height: int, size: int) -> np.ndarray:
        """Create a hexagonal mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        
        for y in range(height):
            for x in range(width):
                # Calculate distance from center using hexagonal metric
                dx = abs(x - center_x)
                dy = abs(y - center_y)
                
                # Hexagonal distance approximation
                hex_dist = max(dx, dy, (dx + dy) // 2)
                
                if hex_dist <= size:
                    mask[y, x] = 1
        
        return mask
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        params = self.get_common_parameters()
        params.update({
            'hex_size': {
                'type': int,
                'default': 8,
                'min': 3,
                'max': 32,
                'description': 'Size of hexagonal cells'
            }
        })
        return params
