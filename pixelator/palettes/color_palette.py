"""
Color palette management for pixel art creation.

This module provides functionality for creating, managing, and applying
color palettes to images for authentic retro video game aesthetics.
"""

import json
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from ..utils.exceptions import InvalidPaletteError


class ColorPalette:
    """Manages color palettes for pixel art conversion."""
    
    def __init__(self, name: str = "Custom", colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        Initialize color palette.
        
        Args:
            name: Name of the palette
            colors: List of RGB color tuples
        """
        self.name = name
        self.colors = colors or []
        
    def add_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """
        Add a color to the palette.
        
        Args:
            color: RGB tuple or hex string
        """
        if isinstance(color, str):
            color = self._hex_to_rgb(color)
        elif not isinstance(color, tuple) or len(color) != 3:
            raise InvalidPaletteError("Color must be RGB tuple or hex string")
            
        if color not in self.colors:
            self.colors.append(color)
    
    def remove_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """
        Remove a color from the palette.
        
        Args:
            color: RGB tuple or hex string
        """
        if isinstance(color, str):
            color = self._hex_to_rgb(color)
            
        if color in self.colors:
            self.colors.remove(color)
    
    def clear(self) -> None:
        """Clear all colors from the palette."""
        self.colors.clear()
    
    def apply_to_image(self, image: Image.Image, dithering: bool = False) -> Image.Image:
        """
        Apply the color palette to an image.
        
        Args:
            image: Input image
            dithering: Whether to apply dithering
            
        Returns:
            Image with applied palette
        """
        if not self.colors:
            raise InvalidPaletteError("Cannot apply empty palette")
            
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Flatten image for easier processing
        pixels = img_array.reshape(-1, 3)
        
        # Find closest palette color for each pixel
        palette_array = np.array(self.colors)
        
        if dithering:
            # Apply Floyd-Steinberg dithering
            new_pixels = self._apply_dithering(img_array, palette_array)
        else:
            # Simple nearest color matching
            new_pixels = self._nearest_color_matching(pixels, palette_array)
        
        # Reshape back to image dimensions
        if dithering:
            result_array = new_pixels
        else:
            result_array = new_pixels.reshape(height, width, 3)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def _nearest_color_matching(self, pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        Find nearest palette color for each pixel using Euclidean distance.
        
        Args:
            pixels: Array of pixel RGB values
            palette: Array of palette RGB values
            
        Returns:
            Array of pixels with palette colors
        """
        # Calculate distances to all palette colors
        distances = np.sqrt(np.sum((pixels[:, np.newaxis] - palette[np.newaxis, :]) ** 2, axis=2))
        
        # Find closest color indices
        closest_indices = np.argmin(distances, axis=1)
        
        # Map to palette colors
        return palette[closest_indices]
    
    def _apply_dithering(self, img_array: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        Apply Floyd-Steinberg dithering while mapping to palette.
        
        Args:
            img_array: Original image array
            palette: Palette colors array
            
        Returns:
            Dithered image array
        """
        height, width = img_array.shape[:2]
        result = img_array.astype(np.float64)
        
        for y in range(height):
            for x in range(width):
                old_pixel = result[y, x]
                
                # Find closest palette color
                distances = np.sqrt(np.sum((old_pixel - palette) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                new_pixel = palette[closest_idx]
                
                result[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < width:
                    result[y, x + 1] += error * 7/16
                if y + 1 < height:
                    if x > 0:
                        result[y + 1, x - 1] += error * 3/16
                    result[y + 1, x] += error * 5/16
                    if x + 1 < width:
                        result[y + 1, x + 1] += error * 1/16
        
        return np.clip(result, 0, 255)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color string to RGB tuple.
        
        Args:
            hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
            
        Returns:
            RGB tuple
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise InvalidPaletteError("Invalid hex color format")
            
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            raise InvalidPaletteError("Invalid hex color format")
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """
        Convert RGB tuple to hex string.
        
        Args:
            rgb: RGB tuple
            
        Returns:
            Hex color string
        """
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save palette to JSON file.
        
        Args:
            file_path: Path to save the palette
        """
        palette_data = {
            'name': self.name,
            'colors': [self._rgb_to_hex(color) for color in self.colors]
        }
        
        with open(file_path, 'w') as f:
            json.dump(palette_data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ColorPalette':
        """
        Load palette from JSON file.
        
        Args:
            file_path: Path to the palette file
            
        Returns:
            ColorPalette instance
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            palette = cls(name=data.get('name', 'Loaded Palette'))
            for hex_color in data.get('colors', []):
                palette.add_color(hex_color)
                
            return palette
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            raise InvalidPaletteError(f"Failed to load palette: {e}")
    
    def generate_preview(self, swatch_size: int = 50) -> Image.Image:
        """
        Generate a preview image showing all colors in the palette.
        
        Args:
            swatch_size: Size of each color swatch
            
        Returns:
            Preview image
        """
        if not self.colors:
            # Return empty image
            return Image.new('RGB', (swatch_size, swatch_size), (255, 255, 255))
        
        cols = min(8, len(self.colors))  # Max 8 columns
        rows = (len(self.colors) + cols - 1) // cols  # Calculate needed rows
        
        width = cols * swatch_size
        height = rows * swatch_size
        
        preview = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(preview)
        
        for i, color in enumerate(self.colors):
            col = i % cols
            row = i // cols
            
            x1 = col * swatch_size
            y1 = row * swatch_size
            x2 = x1 + swatch_size
            y2 = y1 + swatch_size
            
            draw.rectangle([x1, y1, x2-1, y2-1], fill=color)
        
        return preview
    
    def get_dominant_colors(self, image: Image.Image, num_colors: int = 16) -> None:
        """
        Extract dominant colors from an image to create a palette.
        
        Args:
            image: Source image
            num_colors: Number of colors to extract
        """
        # Convert to RGB and resize for faster processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to max 150x150 for performance
        image.thumbnail((150, 150), Image.LANCZOS)
        
        # Convert to array and reshape
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Clear existing colors and add new ones
        self.clear()
        for color in kmeans.cluster_centers_:
            self.add_color(tuple(map(int, color)))
    
    def __len__(self) -> int:
        """Return number of colors in palette."""
        return len(self.colors)
    
    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        """Get color by index."""
        return self.colors[index]
    
    def __str__(self) -> str:
        """String representation of palette."""
        return f"ColorPalette '{self.name}' with {len(self.colors)} colors"


class PredefinedPalettes:
    """Collection of predefined color palettes for classic video game styles."""
    
    @staticmethod
    def get_gameboy() -> ColorPalette:
        """Get classic Game Boy green palette."""
        palette = ColorPalette("Game Boy")
        colors = [
            (15, 56, 15),      # Dark green
            (48, 98, 48),      # Medium green
            (139, 172, 15),    # Light green
            (155, 188, 15)     # Lightest green
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_nes() -> ColorPalette:
        """Get NES-style color palette."""
        palette = ColorPalette("NES")
        colors = [
            (84, 84, 84), (0, 30, 116), (8, 16, 144), (48, 0, 136),
            (68, 0, 100), (92, 0, 48), (84, 4, 0), (60, 24, 0),
            (32, 42, 0), (8, 58, 0), (0, 64, 0), (0, 60, 0),
            (0, 50, 60), (0, 0, 0), (0, 0, 0), (0, 0, 0)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_c64() -> ColorPalette:
        """Get Commodore 64 color palette."""
        palette = ColorPalette("Commodore 64")
        colors = [
            (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
            (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
            (139, 84, 41), (87, 66, 0), (184, 105, 98), (80, 80, 80),
            (120, 120, 120), (148, 224, 137), (120, 105, 196), (159, 159, 159)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_cga() -> ColorPalette:
        """Get CGA color palette."""
        palette = ColorPalette("CGA")
        colors = [
            (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
            (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
            (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
            (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_monochrome() -> ColorPalette:
        """Get monochrome palette."""
        palette = ColorPalette("Monochrome")
        colors = [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_pico8() -> ColorPalette:
        """Get Pico-8 indie game palette."""
        palette = ColorPalette("Pico-8")
        colors = [
            (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
            (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
            (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
            (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_gameboy_color() -> ColorPalette:
        """Get Game Boy Color extended palette."""
        palette = ColorPalette("Game Boy Color")
        colors = [
            (8, 24, 32), (52, 104, 86), (136, 192, 112), (224, 248, 208),
            (64, 40, 32), (144, 88, 64), (224, 168, 120), (248, 224, 200),
            (32, 24, 56), (96, 72, 120), (168, 144, 184), (224, 216, 240),
            (48, 16, 16), (128, 64, 64), (192, 128, 128), (240, 200, 200)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_synthwave() -> ColorPalette:
        """Get synthwave/neon palette."""
        palette = ColorPalette("Synthwave")
        colors = [
            (0, 0, 0), (32, 12, 36), (64, 39, 81), (126, 32, 114),
            (215, 35, 126), (255, 65, 129), (255, 120, 169), (255, 192, 203),
            (16, 16, 30), (35, 39, 85), (66, 126, 137), (109, 194, 202),
            (128, 248, 255), (255, 253, 84), (255, 206, 84), (255, 158, 157)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_earth_tones() -> ColorPalette:
        """Get earth tones palette."""
        palette = ColorPalette("Earth Tones")
        colors = [
            (45, 35, 25), (75, 60, 42), (120, 90, 60), (160, 130, 90),
            (200, 170, 120), (240, 210, 160), (95, 85, 60), (130, 115, 85),
            (165, 145, 110), (200, 180, 140), (85, 70, 50), (115, 95, 70),
            (145, 120, 90), (175, 150, 115), (205, 180, 140), (235, 210, 170)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_pastels() -> ColorPalette:
        """Get pastel color palette."""
        palette = ColorPalette("Pastels")
        colors = [
            (255, 223, 230), (255, 182, 193), (221, 160, 221), (173, 216, 230),
            (176, 224, 230), (175, 238, 238), (152, 251, 152), (240, 230, 140),
            (255, 228, 181), (255, 218, 185), (255, 192, 203), (230, 230, 250),
            (255, 255, 240), (248, 248, 255), (255, 250, 250), (250, 235, 215)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_atari2600() -> ColorPalette:
        """Get Atari 2600 palette."""
        palette = ColorPalette("Atari 2600")
        colors = [
            (0, 0, 0), (64, 64, 64), (108, 108, 108), (144, 144, 144),
            (176, 176, 176), (200, 200, 200), (220, 220, 220), (236, 236, 236),
            (68, 68, 0), (100, 100, 16), (132, 132, 36), (160, 160, 52),
            (184, 184, 64), (208, 208, 80), (232, 232, 92), (252, 252, 104)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    # Mood-based palettes
    @staticmethod
    def get_serene() -> ColorPalette:
        """Get serene/peaceful mood palette."""
        palette = ColorPalette("Serene")
        colors = [
            (32, 46, 84), (73, 106, 156), (143, 185, 224), (224, 243, 255),
            (46, 84, 73), (106, 156, 143), (185, 224, 185), (243, 255, 224),
            (84, 73, 106), (156, 143, 185), (224, 185, 243), (255, 224, 255)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_melancholy() -> ColorPalette:
        """Get melancholy/sad mood palette."""
        palette = ColorPalette("Melancholy")
        colors = [
            (25, 25, 35), (45, 50, 70), (70, 80, 110), (100, 115, 150),
            (35, 30, 45), (55, 50, 75), (80, 75, 115), (115, 110, 155),
            (30, 35, 50), (50, 65, 85), (75, 100, 125), (110, 140, 170)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_energetic() -> ColorPalette:
        """Get energetic/vibrant mood palette."""
        palette = ColorPalette("Energetic")
        colors = [
            (255, 69, 0), (255, 140, 0), (255, 215, 0), (173, 255, 47),
            (0, 191, 255), (138, 43, 226), (255, 20, 147), (255, 105, 180),
            (255, 165, 0), (50, 205, 50), (30, 144, 255), (199, 21, 133)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_mysterious() -> ColorPalette:
        """Get mysterious/dark mood palette."""
        palette = ColorPalette("Mysterious")
        colors = [
            (13, 13, 13), (39, 26, 56), (65, 39, 99), (91, 65, 142),
            (26, 39, 56), (52, 78, 112), (78, 117, 168), (104, 156, 224),
            (39, 26, 39), (78, 52, 78), (117, 78, 117), (156, 104, 156)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_warm_cozy() -> ColorPalette:
        """Get warm/cozy mood palette."""
        palette = ColorPalette("Warm Cozy")
        colors = [
            (139, 69, 19), (160, 82, 45), (205, 133, 63), (222, 184, 135),
            (178, 34, 34), (205, 92, 92), (233, 150, 122), (255, 218, 185),
            (184, 134, 11), (218, 165, 32), (238, 203, 173), (255, 228, 196)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_cool_calm() -> ColorPalette:
        """Get cool/calm mood palette."""
        palette = ColorPalette("Cool Calm")
        colors = [
            (25, 25, 112), (70, 130, 180), (135, 206, 235), (176, 224, 230),
            (46, 139, 87), (102, 205, 170), (175, 238, 238), (240, 248, 255),
            (72, 61, 139), (123, 104, 238), (186, 85, 211), (221, 160, 221)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_dramatic() -> ColorPalette:
        """Get dramatic/intense mood palette."""
        palette = ColorPalette("Dramatic")
        colors = [
            (0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 69, 0),
            (139, 0, 139), (75, 0, 130), (148, 0, 211), (255, 20, 147),
            (25, 25, 25), (169, 169, 169), (255, 255, 255), (255, 215, 0)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_nostalgic() -> ColorPalette:
        """Get nostalgic/vintage mood palette."""
        palette = ColorPalette("Nostalgic")
        colors = [
            (101, 67, 33), (152, 118, 84), (203, 169, 135), (240, 217, 181),
            (139, 90, 43), (190, 141, 94), (218, 192, 163), (245, 231, 201),
            (160, 82, 45), (205, 133, 63), (222, 184, 135), (255, 228, 196)
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    # Lower color resolution palettes
    @staticmethod
    def get_minimal_2bit() -> ColorPalette:
        """Get minimal 2-bit palette (4 colors)."""
        palette = ColorPalette("Minimal 2-bit")
        colors = [
            (0, 0, 0),       # Black
            (85, 85, 85),    # Dark gray
            (170, 170, 170), # Light gray
            (255, 255, 255)  # White
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_gameboy_dmg() -> ColorPalette:
        """Get original Game Boy DMG palette (4 colors)."""
        palette = ColorPalette("Game Boy DMG")
        colors = [
            (8, 24, 32),     # Darkest green
            (52, 104, 86),   # Dark green
            (136, 192, 112), # Light green
            (224, 248, 208)  # Lightest green
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_duochrome_red() -> ColorPalette:
        """Get red duochrome palette (4 colors)."""
        palette = ColorPalette("Duochrome Red")
        colors = [
            (0, 0, 0),       # Black
            (64, 0, 0),      # Dark red
            (128, 0, 0),     # Medium red
            (255, 0, 0)      # Bright red
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_duochrome_blue() -> ColorPalette:
        """Get blue duochrome palette (4 colors)."""
        palette = ColorPalette("Duochrome Blue")
        colors = [
            (0, 0, 0),       # Black
            (0, 0, 64),      # Dark blue
            (0, 0, 128),     # Medium blue
            (0, 100, 255)    # Bright blue
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_vintage_3bit() -> ColorPalette:
        """Get vintage 3-bit style palette (8 colors)."""
        palette = ColorPalette("Vintage 3-bit")
        colors = [
            (0, 0, 0),       # Black
            (128, 0, 0),     # Dark red
            (0, 128, 0),     # Dark green
            (128, 128, 0),   # Dark yellow
            (0, 0, 128),     # Dark blue
            (128, 0, 128),   # Dark magenta
            (0, 128, 128),   # Dark cyan
            (192, 192, 192)  # Light gray
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_amber_mono() -> ColorPalette:
        """Get amber monochrome palette (4 colors)."""
        palette = ColorPalette("Amber Mono")
        colors = [
            (0, 0, 0),       # Black
            (102, 51, 0),    # Dark amber
            (204, 102, 0),   # Medium amber
            (255, 176, 0)    # Bright amber
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_green_mono() -> ColorPalette:
        """Get green monochrome palette (4 colors)."""
        palette = ColorPalette("Green Mono")
        colors = [
            (0, 0, 0),       # Black
            (0, 64, 0),      # Dark green
            (0, 128, 0),     # Medium green
            (0, 255, 0)      # Bright green
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_ice_cream() -> ColorPalette:
        """Get ice cream 8-color palette."""
        palette = ColorPalette("Ice Cream")
        colors = [
            (74, 50, 102),    # Dark purple
            (139, 97, 166),   # Light purple
            (190, 149, 196),  # Pink purple
            (255, 205, 210),  # Light pink
            (255, 239, 186),  # Cream
            (190, 242, 193),  # Mint green
            (139, 207, 226),  # Light blue
            (255, 255, 255)   # White
        ]
        for color in colors:
            palette.add_color(color)
        return palette
    
    @staticmethod
    def get_all_palettes() -> Dict[str, ColorPalette]:
        """Get all predefined palettes."""
        return {
            # Original retro system palettes
            'gameboy': PredefinedPalettes.get_gameboy(),
            'nes': PredefinedPalettes.get_nes(),
            'c64': PredefinedPalettes.get_c64(),
            'cga': PredefinedPalettes.get_cga(),
            'monochrome': PredefinedPalettes.get_monochrome(),
            'pico8': PredefinedPalettes.get_pico8(),
            'gameboy_color': PredefinedPalettes.get_gameboy_color(),
            'synthwave': PredefinedPalettes.get_synthwave(),
            'earth_tones': PredefinedPalettes.get_earth_tones(),
            'pastels': PredefinedPalettes.get_pastels(),
            'atari2600': PredefinedPalettes.get_atari2600(),
            
            # Mood-based palettes
            'serene': PredefinedPalettes.get_serene(),
            'melancholy': PredefinedPalettes.get_melancholy(),
            'energetic': PredefinedPalettes.get_energetic(),
            'mysterious': PredefinedPalettes.get_mysterious(),
            'warm_cozy': PredefinedPalettes.get_warm_cozy(),
            'cool_calm': PredefinedPalettes.get_cool_calm(),
            'dramatic': PredefinedPalettes.get_dramatic(),
            'nostalgic': PredefinedPalettes.get_nostalgic(),
            
            # Lower color resolution palettes
            'minimal_2bit': PredefinedPalettes.get_minimal_2bit(),
            'gameboy_dmg': PredefinedPalettes.get_gameboy_dmg(),
            'duochrome_red': PredefinedPalettes.get_duochrome_red(),
            'duochrome_blue': PredefinedPalettes.get_duochrome_blue(),
            'vintage_3bit': PredefinedPalettes.get_vintage_3bit(),
            'amber_mono': PredefinedPalettes.get_amber_mono(),
            'green_mono': PredefinedPalettes.get_green_mono(),
            'ice_cream': PredefinedPalettes.get_ice_cream()
        }
