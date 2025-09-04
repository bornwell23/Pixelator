"""
Command line interface for the PixelArt Creator.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.image_processor import ImageProcessor
from ..algorithms.manager import AlgorithmManager
from ..palettes.color_palette import ColorPalette, PredefinedPalettes
from ..utils.exceptions import PixelArtError


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert images to pixel art with various algorithms and effects",
        prog="pixelart-creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pixelart-creator input.jpg                                            # Output to output_images/input_pixelated.jpg  
  pixelart-creator input.jpg output.png                                 # Specify exact output file
  pixelart-creator input.jpg my_folder/                                 # Output to my_folder/input_pixelated.jpg
  pixelart-creator input.jpg --target-size 64 64                        # Resize to 64x64 pixels (default behavior)
  pixelart-creator input.jpg --target-size 64 64 --keep-original-size   # Pixelated effect, keep original size
  pixelart-creator input.jpg --demo                                     # Generate multiple demo styles
  pixelart-creator input.jpg --random                                   # Generate 1 random variation
  pixelart-creator input.jpg --random 10                                # Generate 10 random variations
  pixelart-creator "photos/*.jpg" output_folder/                        # Batch process all JPGs in photos folder
  pixelart-creator "*.png" --algorithm nearest --palette gameboy        # Process all PNGs in current folder
  pixelart-creator input.jpg output.png --algorithm edge_preserving --palette gameboy
  pixelart-creator input.jpg output.png --target-size 64 64 --filters contrast:1.2 sharpen:0.5
  pixelart-creator --batch --input-dir ./photos --output-dir ./pixel_art --scale-factor 16
  pixelart-creator --list algorithms
        """
    )
    
    # Add main arguments for single image processing
    add_common_args(parser)
    parser.add_argument("input", nargs="?", help="Input image file path")
    parser.add_argument("output", nargs="?", help="Output image file path (optional - defaults to output_images folder with _pixelated suffix)")
    
    # Add special commands as optional arguments
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--input-dir", help="Input directory for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch processing") 
    parser.add_argument("--pattern", default="*", help="File pattern for batch processing (default: *)")
    
    parser.add_argument("--demo", action="store_true", help="Generate multiple demo styles from input image")
    parser.add_argument("--random", type=int, nargs='?', const=1, default=None, metavar="COUNT", 
                       help="Generate COUNT random pixel art variations from input image (default: 1)")
    
    parser.add_argument("--list", choices=["algorithms", "palettes", "filters"],
                       help="List available algorithms, palettes, or filters")
    
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser."""
    # Size options (mutually exclusive)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--scale-factor", "-s",
        type=float,
        default=8.0,
        help="Pixelation scale factor (default: 8.0)"
    )
    size_group.add_argument(
        "--target-size", "-t",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Target pixelated size in pixels"
    )
    
    # Algorithm selection
    algorithms = AlgorithmManager.get_available_algorithms()
    parser.add_argument(
        "--algorithm", "-a",
        choices=algorithms,
        default="nearest",
        help=f"Pixelation algorithm (default: nearest)"
    )
    
    # Palette options
    predefined_palettes = list(PredefinedPalettes.get_all_palettes().keys())
    parser.add_argument(
        "--palette", "-p",
        help=f"Color palette to apply. Predefined: {', '.join(predefined_palettes)}, or path to custom palette file"
    )
    
    parser.add_argument(
        "--dithering",
        action="store_true",
        help="Apply dithering when using color palettes"
    )
    
    # Filters
    parser.add_argument(
        "--filters", "-f",
        nargs="*",
        help="Filters to apply (format: filter_name:param1:param2)"
    )
    
    # Quality options
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        choices=range(1, 101),
        help="JPEG quality (1-100, default: 95)"
    )
    
    # Preview mode
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview without saving (requires GUI)"
    )
    
    # Resize behavior
    parser.add_argument(
        "--keep-original-size",
        action="store_true",
        help="Keep original image dimensions and apply pixelation as an effect (default: resize to target size)"
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )


def determine_output_path(input_path: str, output_arg: Optional[str]) -> str:
    """
    Determine the output path based on input and output arguments.
    
    Rules:
    1. If no output is given, default to output_images folder with _pixelated suffix
    2. If only folder is given, use same base name with _pixelated suffix in that folder
    3. If only filename is given (no folder), place in output_images folder
    4. If full path is given, use as-is
    """
    input_path_obj = Path(input_path)
    input_stem = input_path_obj.stem
    input_suffix = input_path_obj.suffix
    
    # Case 1: No output argument given
    if not output_arg:
        output_dir = Path("output_images")
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / f"{input_stem}_pixelated{input_suffix}")
    
    output_path_obj = Path(output_arg)
    
    # Case 2: Output is a directory (ends with / or is an existing directory)
    if output_arg.endswith(('/', '\\')) or (output_path_obj.exists() and output_path_obj.is_dir()):
        output_dir = output_path_obj
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{input_stem}_pixelated{input_suffix}")
    
    # Case 3: Output is just a filename (no directory separators)
    if '/' not in output_arg and '\\' not in output_arg:
        output_dir = Path("output_images")
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / output_arg)
    
    # Case 4: Full path given - use as-is, but ensure directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    return output_arg


def process_single_image(args) -> None:
    """Process a single image."""
    if not args.input:
        print("Error: Input path is required for single image processing")
        sys.exit(1)
    
    # Handle default output logic
    output_path = determine_output_path(args.input, args.output)
    
    try:
        # Initialize processor
        processor = ImageProcessor()
        
        if args.verbose:
            print(f"Loading image: {args.input}")
        
        # Load image
        processor.load_image(args.input)
        
        # Get algorithm
        algorithm = AlgorithmManager.get_algorithm(args.algorithm)
        if not algorithm:
            print(f"Error: Algorithm '{args.algorithm}' not found")
            sys.exit(1)
        
        # Determine target size
        if args.target_size:
            target_size = tuple(args.target_size)
        else:
            original_size = processor.current_image.size
            target_size = (
                int(original_size[0] / args.scale_factor),
                int(original_size[1] / args.scale_factor)
            )
        
        if args.verbose:
            print(f"Pixelating with {args.algorithm} algorithm to size {target_size}")
        
        # Apply pixelation
        processor.pixelate(algorithm, target_size, resize_to_target=not args.keep_original_size)
        
        # Apply palette if specified
        if args.palette:
            palette = load_palette(args.palette)
            if palette:
                if args.verbose:
                    print(f"Applying palette: {palette.name}")
                processor.apply_palette(palette)
        
        # Apply filters if specified
        if args.filters:
            apply_filters(processor, args.filters, args.verbose)
        
        # Save result
        if args.verbose:
            print(f"Saving to: {output_path}")
        
        processor.save_image(output_path, quality=args.quality)
        
        if args.verbose:
            info = processor.get_image_info()
            print(f"Complete! Final size: {info['size']}, mode: {info['mode']}")
        
    except PixelArtError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def process_batch(args) -> None:
    """Process multiple images in batch."""
    from glob import glob
    import os
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    pattern = args.pattern if args.pattern else "*"
    search_pattern = input_dir / pattern
    input_files = glob(str(search_pattern))
    
    # Filter for image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    input_files = [f for f in input_files if Path(f).suffix.lower() in image_extensions]
    
    if not input_files:
        print(f"No image files found in '{input_dir}' matching pattern '{pattern}'")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(input_files)} images to process")
    
    # Process each file
    processed = 0
    errors = 0
    
    for input_file in input_files:
        try:
            input_path = Path(input_file)
            output_path = output_dir / f"{input_path.stem}_pixelated{input_path.suffix}"
            
            if args.verbose:
                print(f"Processing: {input_path.name}")
            
            # Create temporary args for single image processing
            temp_args = argparse.Namespace(**vars(args))
            temp_args.input = str(input_path)
            temp_args.output = str(output_path)
            
            process_single_image(temp_args)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            errors += 1
    
    print(f"Batch processing complete. Processed: {processed}, Errors: {errors}")


def list_items(list_type: str) -> None:
    """List available algorithms, palettes, or filters."""
    if list_type == "algorithms":
        algorithms = AlgorithmManager.get_algorithm_info()
        print("Available pixelation algorithms:")
        for name, description in algorithms.items():
            print(f"  {name:15} - {description}")
    
    elif list_type == "palettes":
        palettes = PredefinedPalettes.get_all_palettes()
        print("Available predefined palettes:")
        for name, palette in palettes.items():
            print(f"  {name:15} - {len(palette)} colors")
    
    elif list_type == "filters":
        from ..filters.image_filters import ImageFilters
        filters = ImageFilters.get_available_filters()
        print("Available image filters:")
        for name, description in filters.items():
            print(f"  {name:20} - {description}")


def load_palette(palette_name: str) -> Optional[ColorPalette]:
    """Load a color palette by name or file path."""
    # Check if it's a predefined palette
    predefined = PredefinedPalettes.get_all_palettes()
    if palette_name.lower() in predefined:
        return predefined[palette_name.lower()]
    
    # Try to load from file
    palette_path = Path(palette_name)
    if palette_path.exists():
        try:
            return ColorPalette.load_from_file(palette_path)
        except Exception as e:
            print(f"Warning: Failed to load palette from '{palette_path}': {e}")
    
    print(f"Warning: Palette '{palette_name}' not found")
    return None


def apply_filters(processor: ImageProcessor, filters: list, verbose: bool = False) -> None:
    """Apply filters to the processor."""
    # Define parameter names for each filter
    filter_params = {
        'blur': ['radius'],
        'sharpen': ['factor'],
        'contrast': ['factor'],
        'brightness': ['factor'],
        'saturation': ['factor'],
        'posterize': ['bits'],
        'solarize': ['threshold'],
        'sepia': ['intensity'],
        'noise': ['intensity'],
        'vignette': ['intensity'],
        'scan_lines': ['line_height', 'opacity'],
        'chromatic_aberration': ['offset'],
        'pixelate_mosaic': ['block_size'],
        'halftone': ['dot_size'],
        'cross_hatch': ['line_density'],
        'oil_painting': ['radius', 'intensity'],
        'film_grain': ['intensity', 'grain_size'],
        'color_shift': ['hue_shift', 'saturation_factor', 'value_factor'],
        'texture_overlay': ['texture_type', 'intensity']
    }
    
    for filter_spec in filters:
        parts = filter_spec.split(':')
        filter_name = parts[0]
        
        # Parse parameters
        kwargs = {}
        param_names = filter_params.get(filter_name, ['factor', 'intensity', 'radius', 'threshold', 'strength'])
        
        for i, param in enumerate(parts[1:]):
            try:
                # Try to convert to number
                if '.' in param:
                    value = float(param)
                else:
                    value = int(param)
            except ValueError:
                # Keep as string
                value = param
            
            # Use correct parameter name for this filter
            if i < len(param_names):
                kwargs[param_names[i]] = value
        
        try:
            if verbose:
                print(f"Applying filter: {filter_name} with params {kwargs}")
            processor.apply_filter(filter_name, **kwargs)
        except Exception as e:
            print(f"Warning: Failed to apply filter '{filter_name}': {e}")


def process_demo(args) -> None:
    """Generate multiple demo styles from a single input image."""
    if not args.input:
        print("Error: Input path is required for demo mode")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Create demo output directory
    demo_dir = Path("demo_output")
    demo_dir.mkdir(exist_ok=True)
    
    base_name = input_path.stem
    
    # Define demo styles
    demo_styles = [
        {
            "name": "gameboy_classic",
            "description": "Classic Game Boy green screen",
            "algorithm": "edge_preserving",
            "target_size": (64, 64),
            "palette": "gameboy",
            "dithering": True,
            "filters": ["contrast:1.2", "brightness:0.9"]
        },
        {
            "name": "nes_retro",
            "description": "NES-style 8-bit colors",
            "algorithm": "nearest",
            "target_size": (80, 80),
            "palette": "nes",
            "dithering": False,
            "filters": ["contrast:1.1", "saturation:1.2"]
        },
        {
            "name": "c64_nostalgic",
            "description": "Commodore 64 color palette",
            "algorithm": "bilinear",
            "target_size": (96, 96),
            "palette": "c64",
            "dithering": True,
            "filters": ["contrast:1.0"]
        },
        {
            "name": "crt_monitor",
            "description": "CRT monitor with scan lines",
            "algorithm": "bilinear",
            "scale_factor": 8,
            "keep_original_size": True,
            "filters": ["scan_lines:3:0.4", "vignette:0.5", "chromatic_aberration:2"]
        },
        {
            "name": "high_quality",
            "description": "High-quality Lanczos pixelation",
            "algorithm": "lanczos",
            "target_size": (128, 128),
            "filters": ["edge_enhance", "sharpen:0.3", "contrast:1.1"]
        },
        {
            "name": "monochrome_art",
            "description": "Dramatic black and white",
            "algorithm": "edge_preserving",
            "target_size": (64, 64),
            "palette": "monochrome",
            "dithering": True,
            "filters": ["contrast:2.0", "brightness:0.8"]
        },
        {
            "name": "sepia_vintage",
            "description": "Vintage sepia-toned pixel art",
            "algorithm": "lanczos",
            "scale_factor": 6,
            "keep_original_size": True,
            "filters": ["sepia:0.8", "vignette:0.3", "contrast:1.1"]
        },
        {
            "name": "super_pixel_art",
            "description": "Super pixel clustering effect",
            "algorithm": "super_pixel",
            "target_size": (80, 80),
            "filters": ["posterize:6", "saturation:1.3", "contrast:1.1"]
        }
    ]
    
    print(f"Generating {len(demo_styles)} demo styles from '{input_path}'...")
    print(f"Output directory: {demo_dir}")
    
    successful = 0
    failed = 0
    
    for style in demo_styles:
        try:
            output_file = demo_dir / f"{base_name}_{style['name']}.png"
            
            if args.verbose:
                print(f"\nCreating: {style['name']} - {style['description']}")
                print(f"  Output: {output_file}")
            
            # Initialize processor
            processor = ImageProcessor()
            processor.load_image(args.input)
            
            # Get algorithm
            algorithm = AlgorithmManager.get_algorithm(style["algorithm"])
            if not algorithm:
                print(f"Warning: Algorithm '{style['algorithm']}' not found, skipping {style['name']}")
                failed += 1
                continue
            
            # Determine target size
            if "target_size" in style:
                target_size = style["target_size"]
                keep_original_size = style.get("keep_original_size", False)
            elif "scale_factor" in style:
                original_size = processor.current_image.size
                scale_factor = style["scale_factor"]
                target_size = (
                    int(original_size[0] / scale_factor),
                    int(original_size[1] / scale_factor)
                )
                keep_original_size = style.get("keep_original_size", False)
            else:
                # Default fallback
                target_size = (64, 64)
                keep_original_size = False
            
            # Apply pixelation
            processor.pixelate(algorithm, target_size, resize_to_target=not keep_original_size)
            
            # Apply palette if specified
            if "palette" in style:
                palette = load_palette(style["palette"])
                if palette:
                    if args.verbose:
                        print(f"  Applying palette: {style['palette']}")
                    
                    # Apply dithering if specified
                    if style.get("dithering", False):
                        palette.dithering = True
                    
                    processor.apply_palette(palette)
            
            # Apply filters if specified
            if "filters" in style:
                apply_filters(processor, style["filters"], args.verbose)
            
            # Save result
            processor.save_image(output_file, quality=95)
            
            if args.verbose:
                info = processor.get_image_info()
                print(f"  Saved: {info['size']}, mode: {info['mode']}")
            else:
                print(f"✓ {style['name']}")
                
            successful += 1
            
        except Exception as e:
            print(f"✗ Failed to create {style['name']}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
    
    print(f"\nDemo generation complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Output files saved in: {demo_dir}")


def process_random(args) -> None:
    """Generate random pixel art variations from a single input image."""
    import random
    import os
    
    if not args.input:
        print("Error: Input path is required for random mode")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    count = args.random
    if count <= 0:
        print("Error: Count must be positive")
        sys.exit(1)
    
    # Create random output directory
    random_dir = Path("random_output")
    random_dir.mkdir(exist_ok=True)
    
    base_name = input_path.stem
    
    # Define available options for randomization
    algorithms = ["nearest", "bilinear", "lanczos", "edge_preserving", "super_pixel", "adaptive", 
                 "quantization", "voronoi", "hexagonal"]
    palettes = ["gameboy", "nes", "c64", "monochrome", "pico8", "gameboy_color", "synthwave", 
               "earth_tones", "pastels", "atari2600"]
    
    # Define possible filter combinations
    available_filters = {
        "contrast": (0.5, 2.0),      # (min, max)
        "brightness": (0.6, 1.4),
        "saturation": (0.5, 2.0),
        "blur": (0.5, 2.0),
        "sharpen": (0.1, 1.0),
        "posterize": (2, 8),         # bits
        "noise": (0.05, 0.3),        # intensity
        "sepia": (0.3, 1.0),         # intensity
        "vignette": (0.2, 0.8),      # intensity
        "halftone": (3, 8),          # dot_size
        "cross_hatch": (4, 12),      # line_density
        "oil_painting": (2, 6),      # radius (intensity will be auto-set)
        "film_grain": (0.05, 0.2),  # intensity
    }
    
    print(f"Generating {count} random pixel art variations from '{input_path}'...")
    print(f"Output directory: {random_dir}")
    
    successful = 0
    failed = 0
    
    for i in range(count):
        try:
            # Randomize parameters
            algorithm = random.choice(algorithms)
            palette = random.choice(palettes + [None]) if random.random() < 0.8 else None
            
            # Random size - either scale factor or target size
            if random.choice([True, False]):
                # Use scale factor
                scale_factor = random.randint(1, 12)
                target_size = None
                keep_original_size = random.choice([True, False])
                size_desc = f"scale{scale_factor}"
                if keep_original_size:
                    size_desc += "_keepsize"
            else:
                # Use target size
                size = random.choice([32, 48, 64, 80, 96, 128, 256, 512])
                target_size = (size, size)
                keep_original_size = False
                size_desc = f"{size}x{size}"
            
            # Randomize dithering
            dithering = random.choice([True, False]) if palette else False
            
            # Random filters (0-3 filters)
            num_filters = random.choices([0, 1, 2, 3], weights=[0.4, 0.4, 0.15, 0.05])[0]
            filters = []
            filter_names = random.sample(list(available_filters.keys()), 
                                       min(num_filters, len(available_filters)))
            
            filter_desc_parts = []
            for filter_name in filter_names:
                min_val, max_val = available_filters[filter_name]
                if filter_name == "posterize":
                    value = random.randint(int(min_val), int(max_val))
                else:
                    value = round(random.uniform(min_val, max_val), 2)
                filters.append(f"{filter_name}:{value}")
                filter_desc_parts.append(f"{filter_name}{value}")
            
            # Build descriptive filename
            filename_parts = [base_name, algorithm]
            if palette:
                filename_parts.append(palette)
            if dithering:
                filename_parts.append("dither")
            filename_parts.append(size_desc)
            if filter_desc_parts:
                filename_parts.extend(filter_desc_parts)
            
            # Limit filename length and make it filesystem-safe
            filename = "_".join(filename_parts)
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            if len(filename) > 200:  # Limit length
                filename = filename[:200]
            
            output_file = random_dir / f"{filename}.png"
            
            # Print parameters before processing
            print(f"\n[{i+1}/{count}] Creating variation with parameters:")
            print(f"  Algorithm: {algorithm}")
            print(f"  Palette: {palette or 'None'}")
            print(f"  Dithering: {dithering}")
            print(f"  Size: {size_desc}")
            print(f"  Filters: {filters or 'None'}")
            print(f"  Output: {output_file.name}")
            
            if args.verbose:
                print(f"  Processing...")
            
            # Initialize processor
            processor = ImageProcessor()
            processor.load_image(args.input)
            
            # Get algorithm
            algorithm_obj = AlgorithmManager.get_algorithm(algorithm)
            if not algorithm_obj:
                print(f"Warning: Algorithm '{algorithm}' not found, skipping variation {i+1}")
                failed += 1
                continue
            
            # Apply pixelation
            if target_size:
                processor.pixelate(algorithm_obj, target_size, resize_to_target=not keep_original_size)
            else:
                # Calculate target size from scale factor
                original_size = processor.current_image.size
                calc_target_size = (
                    int(original_size[0] / scale_factor),
                    int(original_size[1] / scale_factor)
                )
                processor.pixelate(algorithm_obj, calc_target_size, resize_to_target=not keep_original_size)
            
            # Apply palette if specified
            if palette:
                palette_obj = load_palette(palette)
                if palette_obj:
                    if dithering:
                        palette_obj.dithering = True
                    processor.apply_palette(palette_obj)
            
            # Apply filters if specified
            if filters:
                apply_filters(processor, filters, args.verbose)
            
            # Save result
            processor.save_image(output_file, quality=95)
            
            if args.verbose:
                info = processor.get_image_info()
                print(f"  ✓ Saved: {info['size']}, mode: {info['mode']}")
            else:
                print(f"  ✓ Successfully created!")
                
            successful += 1
            
        except Exception as e:
            print(f"✗ [{i+1}/{count}] Failed to create variation: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
    
    print(f"\nRandom generation complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Output files saved in: {random_dir}")


def process_wildcard_batch(args) -> None:
    """Process multiple images using wildcard patterns in the input path."""
    import glob
    import os
    from pathlib import Path
    
    # Expand the wildcard pattern
    input_pattern = args.input
    matched_files = glob.glob(input_pattern)
    
    if not matched_files:
        print(f"No files found matching pattern: {input_pattern}")
        sys.exit(1)
    
    # Filter for image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    image_files = [f for f in matched_files if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found matching pattern: {input_pattern}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image files matching pattern: {input_pattern}")
    
    # Determine output strategy
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or args.output.endswith('/') or args.output.endswith('\\'):
            # Output is a directory
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            use_output_dir = True
        else:
            # Output is a specific file - only works with single input
            if len(image_files) > 1:
                print("Error: Cannot specify a single output file for multiple input files")
                print("Use a directory path for the output instead")
                sys.exit(1)
            use_output_dir = False
    else:
        # No output specified - use default output_images directory
        output_dir = Path("output_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        use_output_dir = True
    
    if args.verbose:
        print(f"Processing {len(image_files)} files...")
        if use_output_dir:
            print(f"Output directory: {output_dir}")
    
    # Process each file
    processed = 0
    errors = 0
    
    for input_file in image_files:
        try:
            input_path = Path(input_file)
            
            if use_output_dir:
                # Generate output filename in the output directory
                output_file = output_dir / f"{input_path.stem}_pixelated{input_path.suffix}"
            else:
                # Use the specific output file
                output_file = Path(args.output)
            
            if args.verbose:
                print(f"Processing: {input_path.name} -> {output_file.name}")
            
            # Create temporary args for single image processing
            temp_args = argparse.Namespace(**vars(args))
            temp_args.input = str(input_path)
            temp_args.output = str(output_file)
            
            # Process the single image
            process_single_image(temp_args)
            processed += 1
            
            if not args.verbose:
                print(f"✓ {input_path.name}")
            
        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            errors += 1
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {processed}")
    if errors > 0:
        print(f"Errors: {errors}")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle different commands
    if args.list:
        list_items(args.list)
    elif args.demo:
        if not args.input:
            print("Error: Input file is required for demo mode")
            sys.exit(1)
        process_demo(args)
    elif args.random:
        if not args.input:
            print("Error: Input file is required for random mode")
            sys.exit(1)
        process_random(args)
    elif args.batch:
        if not args.input_dir or not args.output_dir:
            print("Error: --input-dir and --output-dir are required for batch processing")
            sys.exit(1)
        process_batch(args)
    elif args.input:
        # Check if input contains wildcards for batch processing
        if '*' in args.input or '?' in args.input:
            # Handle wildcard batch processing
            process_wildcard_batch(args)
        else:
            # Handle direct input (output is optional now)
            process_single_image(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
