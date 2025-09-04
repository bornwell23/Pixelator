"""
Custom exceptions for the PixelArt Creator application.
"""


class PixelArtError(Exception):
    """Base exception for PixelArt Creator errors."""
    pass


class UnsupportedFormatError(PixelArtError):
    """Raised when an unsupported image format is encountered."""
    pass


class InvalidAlgorithmError(PixelArtError):
    """Raised when an invalid pixelation algorithm is specified."""
    pass


class InvalidPaletteError(PixelArtError):
    """Raised when an invalid color palette is encountered."""
    pass


class ProcessingError(PixelArtError):
    """Raised when image processing fails."""
    pass


class ConfigurationError(PixelArtError):
    """Raised when there's a configuration error."""
    pass


class BatchProcessingError(PixelArtError):
    """Raised when batch processing fails."""
    pass
