"""
Configuration management for the PixelArt Creator application.
"""

from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PixelationConfig:
    """Configuration for pixelation settings."""
    default_scale_factor: int = 8
    default_method: str = "nearest_neighbor"
    preserve_aspect_ratio: bool = True
    max_image_size: tuple = (8192, 8192)
    default_palette_size: int = 16
    enable_dithering: bool = False


@dataclass
class UIConfig:
    """Configuration for user interface settings."""
    window_width: int = 1200
    window_height: int = 800
    preview_size: tuple = (400, 400)
    auto_preview: bool = True
    remember_last_directory: bool = True
    last_input_directory: str = ""
    last_output_directory: str = ""


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    max_preview_size: tuple = (800, 800)
    chunk_size: int = 1024
    enable_multiprocessing: bool = True
    max_workers: int = 4


class ConfigManager:
    """
    Manages application configuration and user preferences.
    
    Handles loading, saving, and validation of configuration settings
    with support for user overrides and defaults.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        if config_dir is None:
            # Use platform-appropriate config directory
            if Path.home().exists():
                config_dir = Path.home() / ".pixelart_creator"
            else:
                config_dir = Path.cwd() / "config"
                
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.json"
        
        # Configuration objects
        self.pixelation = PixelationConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.config_file.exists():
                logger.info("No configuration file found, using defaults")
                return True
                
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                
            # Load each configuration section
            if 'pixelation' in data:
                self._update_config(self.pixelation, data['pixelation'])
                
            if 'ui' in data:
                self._update_config(self.ui, data['ui'])
                
            if 'performance' in data:
                self._update_config(self.performance, data['performance'])
                
            logger.info(f"Loaded configuration from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_data = {
                'pixelation': asdict(self.pixelation),
                'ui': asdict(self.ui),
                'performance': asdict(self.performance)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Saved configuration to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        self.pixelation = PixelationConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        logger.info("Reset configuration to defaults")
    
    def get_pixelation_params(self) -> Dict[str, Any]:
        """
        Get pixelation parameters as dictionary.
        
        Returns:
            Dictionary of pixelation parameters
        """
        return {
            'scale_factor': self.pixelation.default_scale_factor,
            'preserve_aspect_ratio': self.pixelation.preserve_aspect_ratio,
            'palette_colors': self.pixelation.default_palette_size,
            'dithering': self.pixelation.enable_dithering
        }
    
    def update_last_directories(self, input_dir: str = None, output_dir: str = None) -> None:
        """
        Update last used directories.
        
        Args:
            input_dir: Last input directory
            output_dir: Last output directory
        """
        if input_dir and self.ui.remember_last_directory:
            self.ui.last_input_directory = input_dir
            
        if output_dir and self.ui.remember_last_directory:
            self.ui.last_output_directory = output_dir
    
    def validate_config(self) -> Dict[str, list]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {
            'pixelation': [],
            'ui': [],
            'performance': []
        }
        
        # Validate pixelation config
        if self.pixelation.default_scale_factor < 1:
            errors['pixelation'].append("Scale factor must be >= 1")
            
        if self.pixelation.default_palette_size < 2:
            errors['pixelation'].append("Palette size must be >= 2")
            
        # Validate UI config
        if self.ui.window_width < 400:
            errors['ui'].append("Window width must be >= 400")
            
        if self.ui.window_height < 300:
            errors['ui'].append("Window height must be >= 300")
            
        # Validate performance config
        if self.performance.max_workers < 1:
            errors['performance'].append("Max workers must be >= 1")
            
        if self.performance.chunk_size < 64:
            errors['performance'].append("Chunk size must be >= 64")
        
        return {k: v for k, v in errors.items() if v}
    
    def _update_config(self, config_obj: Any, data: Dict[str, Any]) -> None:
        """
        Update configuration object with data from dictionary.
        
        Args:
            config_obj: Configuration object to update
            data: Dictionary with new values
        """
        for key, value in data.items():
            if hasattr(config_obj, key):
                # Handle tuple conversion for tuple fields
                current_value = getattr(config_obj, key)
                if isinstance(current_value, tuple) and isinstance(value, list):
                    value = tuple(value)
                    
                setattr(config_obj, key, value)
    
    def export_config(self, export_path: Path) -> bool:
        """
        Export configuration to a file.
        
        Args:
            export_path: Path to export configuration
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            config_data = {
                'pixelation': asdict(self.pixelation),
                'ui': asdict(self.ui),
                'performance': asdict(self.performance)
            }
            
            with open(export_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Exported configuration to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """
        Import configuration from a file.
        
        Args:
            import_path: Path to import configuration from
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
                
            # Backup current config
            backup_pixelation = self.pixelation
            backup_ui = self.ui
            backup_performance = self.performance
            
            try:
                # Load each section
                if 'pixelation' in data:
                    self._update_config(self.pixelation, data['pixelation'])
                    
                if 'ui' in data:
                    self._update_config(self.ui, data['ui'])
                    
                if 'performance' in data:
                    self._update_config(self.performance, data['performance'])
                    
                # Validate imported config
                errors = self.validate_config()
                if any(errors.values()):
                    raise ValueError(f"Invalid configuration: {errors}")
                    
                logger.info(f"Imported configuration from {import_path}")
                return True
                
            except Exception:
                # Restore backup on error
                self.pixelation = backup_pixelation
                self.ui = backup_ui
                self.performance = backup_performance
                raise
                
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
