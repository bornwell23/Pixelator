"""
Command-line interface for PixelArt Creator.
"""

import click
import logging
from pathlib import Path
from typing import Optional

from ..core.processor import ImageProcessor
from ..core.pixelator import Pixelator, PixelationMethod
from ..core.config import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(), help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose, config):
    """PixelArt Creator - Convert images to pixel art."""
    ctx.ensure_object(dict)
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize configuration
    config_manager = ConfigManager()
    if config:
        config_manager.import_config(Path(config))
    
    ctx.obj['config'] = config_manager


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path (default: input_pixelated.ext)')
@click.option('--scale', '-s', type=int, default=8,
              help='Pixelation scale factor (default: 8)')
@click.option('--method', '-m', 
              type=click.Choice(['nearest_neighbor', 'bilinear', 'custom']),
              default='nearest_neighbor',
              help='Pixelation method')
@click.option('--palette-size', type=int, default=16,
              help='Number of colors in palette (default: 16)')
@click.option('--no-aspect-ratio', is_flag=True,
              help='Do not preserve aspect ratio')
@click.option('--preview', is_flag=True,
              help='Show preview without saving')
@click.pass_context
def pixelate(ctx, input_path, output, scale, method, palette_size, 
             no_aspect_ratio, preview):
    """Pixelate a single image."""
    
    config = ctx.obj['config']
    
    # Initialize processor and pixelator
    processor = ImageProcessor()
    pixelator = Pixelator(processor)
    
    # Load image
    click.echo(f"Loading image: {input_path}")
    if not processor.load_image(input_path):
        click.echo("Error: Failed to load image", err=True)
        return
    
    # Get image info
    info = processor.get_image_info()
    click.echo(f"Image size: {info['size'][0]}x{info['size'][1]}")
    click.echo(f"Image mode: {info['mode']}")
    
    # Set up parameters
    params = {
        'scale_factor': scale,
        'preserve_aspect_ratio': not no_aspect_ratio,
        'palette_colors': palette_size
    }
    
    # Apply pixelation
    click.echo(f"Applying {method} pixelation (scale: {scale})...")
    
    pixelation_method = PixelationMethod(method)
    result = pixelator.pixelate(pixelation_method, **params)
    
    if result is None:
        click.echo("Error: Pixelation failed", err=True)
        return
    
    final_size = result.size
    click.echo(f"Final size: {final_size[0]}x{final_size[1]}")
    
    if preview:
        # Show preview (would open image viewer)
        click.echo("Preview mode - image processed but not saved")
        try:
            result.show()
        except Exception:
            click.echo("Cannot display preview on this system")
        return
    
    # Determine output path
    if not output:
        input_file = Path(input_path)
        output = input_file.parent / f"{input_file.stem}_pixelated{input_file.suffix}"
    
    # Save result
    click.echo(f"Saving to: {output}")
    if processor.save_image(output, result):
        click.echo("✓ Successfully saved pixelated image")
    else:
        click.echo("Error: Failed to save image", err=True)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('--scale', '-s', type=int, default=8,
              help='Pixelation scale factor (default: 8)')
@click.option('--method', '-m',
              type=click.Choice(['nearest_neighbor', 'bilinear', 'custom']),
              default='nearest_neighbor',
              help='Pixelation method')
@click.option('--palette-size', type=int, default=16,
              help='Number of colors in palette (default: 16)')
@click.option('--pattern', default='*.{png,jpg,jpeg,bmp,gif,tiff}',
              help='File pattern to match (default: common image formats)')
@click.option('--no-aspect-ratio', is_flag=True,
              help='Do not preserve aspect ratio')
@click.pass_context
def batch(ctx, input_dir, output_dir, scale, method, palette_size, 
          pattern, no_aspect_ratio):
    """Batch process multiple images."""
    
    config = ctx.obj['config']
    
    # Find input files
    input_path = Path(input_dir)
    patterns = pattern.split(',')
    input_files = []
    
    for pat in patterns:
        input_files.extend(input_path.glob(pat.strip()))
    
    if not input_files:
        click.echo(f"No files found matching pattern: {pattern}", err=True)
        return
    
    click.echo(f"Found {len(input_files)} files to process")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor and pixelator
    processor = ImageProcessor()
    pixelator = Pixelator(processor)
    
    # Set up parameters
    params = {
        'scale_factor': scale,
        'preserve_aspect_ratio': not no_aspect_ratio,
        'palette_colors': palette_size
    }
    
    # Process files
    pixelation_method = PixelationMethod(method)
    
    with click.progressbar(input_files, label='Processing images') as files:
        successful = 0
        failed = 0
        
        for file_path in files:
            try:
                # Load image
                if not processor.load_image(file_path):
                    failed += 1
                    continue
                
                # Apply pixelation
                result = pixelator.pixelate(pixelation_method, **params)
                
                if result is None:
                    failed += 1
                    continue
                
                # Save result
                output_file = output_path / f"{file_path.stem}_pixelated{file_path.suffix}"
                
                if processor.save_image(output_file, result):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed += 1
    
    click.echo(f"\nBatch processing complete:")
    click.echo(f"  Successful: {successful}")
    click.echo(f"  Failed: {failed}")


@cli.command()
@click.pass_context
def gui(ctx):
    """Launch the graphical user interface."""
    click.echo("Launching GUI...")
    
    try:
        from ..gui.main_window import PixelArtGUI
        app = PixelArtGUI()
        app.run()
    except ImportError:
        click.echo("Error: GUI dependencies not available", err=True)
    except Exception as e:
        click.echo(f"Error launching GUI: {e}", err=True)


@cli.command()
@click.pass_context
def config_info(ctx):
    """Show current configuration."""
    config = ctx.obj['config']
    
    click.echo("Current Configuration:")
    click.echo(f"  Config file: {config.config_file}")
    click.echo(f"  Default scale factor: {config.pixelation.default_scale_factor}")
    click.echo(f"  Default method: {config.pixelation.default_method}")
    click.echo(f"  Max image size: {config.pixelation.max_image_size}")
    click.echo(f"  UI preview size: {config.ui.preview_size}")
    click.echo(f"  Performance workers: {config.performance.max_workers}")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
