"""
PixelArt Creator - A Python application for converting images into pixel art.

This package provides both CLI and GUI interfaces for creating pixel art from images
with various algorithms, color palettes, and effects.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.processor import ImageProcessor
from .core.pixelator import Pixelator

__all__ = ["ImageProcessor", "Pixelator"]
