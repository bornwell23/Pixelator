# PixelArt Creator

A comprehensive Python application for converting images into pixel art with various algorithms, color palettes, and effects. Supports both command-line and GUI interfaces.

## Features

### Core Functionality
- **Multiple Pixelation Algorithms**
  - Nearest Neighbor (simple and fast)
  - Bilinear (smoother downsampling)
  - Lanczos (high-quality downsampling)
  - Edge Preserving (maintains important boundaries)
  - Super Pixel (groups similar pixels)
  - Adaptive (variable pixelation based on image complexity)

- **Color Palette Support**
  - Predefined palettes (Game Boy, NES, C64, CGA, Monochrome)
  - Custom palette loading from JSON files
  - Optional dithering for smooth color transitions
  - Palette extraction from images

- **Image Filters & Effects**
  - Basic adjustments (contrast, brightness, saturation)
  - Artistic effects (sepia, emboss, solarize)
  - Retro effects (scan lines, chromatic aberration, CRT simulation)
  - Noise and blur filters
  - Edge detection and enhancement

- **Batch Processing**
  - Process multiple images at once
  - Customizable file patterns
  - Progress tracking and error handling

- **Live Preview**
  - Real-time preview in GUI
  - Non-destructive editing
  - Easy comparison with original

### Interface Options
- **Command Line Interface (CLI)** - Fast and scriptable
- **Graphical User Interface (GUI)** - User-friendly with live preview

## Installation

### Requirements
- Python 3.8 or higher
- Required packages (automatically installed):
  - Pillow (image processing)
  - NumPy (numerical operations)
  - OpenCV (computer vision)
  - scikit-image (advanced image processing)
  - scikit-learn (machine learning for color clustering)
  - scipy (scientific computing)
  - click (CLI interface)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Understanding Target Size vs Resize Behavior

**Important:** By default, `--target-size` now **resizes the image** to the specified dimensions:

```bash
# This RESIZES the image to 64x64 pixels (default behavior)
python -m pixelart_creator 800x600_photo.jpg small_sprite.png --target-size 64 64
# Result: 64x64 pixel image

# This creates a pixelated EFFECT but keeps original image size  
python -m pixelart_creator 800x600_photo.jpg pixelated.png --target-size 64 64 --keep-original-size
# Result: 800x600 image with 64x64 pixel blocks
```

**Use Cases:**
- **True Resize** (default): Game sprites, icons, thumbnails, memory-constrained applications
- **Pixelated Effect** (`--keep-original-size`): Social media filters, artistic effects, retro styling

### Command Line Interface

#### Demo Mode

The `--demo` command generates multiple styles from a single input image, perfect for showcasing the different capabilities:

```bash
# Generate 8 different pixel art styles from one image
python -m pixelart_creator photo.jpg --demo

# Demo with verbose output to see what's being generated
python -m pixelart_creator photo.jpg --demo --verbose
```

**Generated Demo Styles:**
- **gameboy_classic** - Classic Game Boy green screen effect
- **nes_retro** - NES-style 8-bit colors with crisp pixels
- **c64_nostalgic** - Commodore 64 color palette with dithering
- **crt_monitor** - CRT monitor simulation with scan lines and effects
- **high_quality** - High-quality Lanczos pixelation
- **monochrome_art** - Dramatic black and white pixel art
- **sepia_vintage** - Vintage sepia-toned effect
- **super_pixel_art** - Super pixel clustering with posterization

All demo outputs are saved in the `demo_output/` folder with descriptive filenames.

#### Basic Usage
```bash
# Simple pixelation with scale factor
python -m pixelart_creator input.jpg output.png --scale-factor 8

# Specify exact target size
python -m pixelart_creator input.jpg output.png --target-size 64 64

# Keep original size with pixelated effect
python -m pixelart_creator input.jpg output.png --target-size 64 64 --keep-original-size

# Generate multiple demo styles from one image
python -m pixelart_creator input.jpg --demo

# Use different algorithm
python -m pixelart_creator input.jpg output.png --algorithm edge_preserving

# Apply color palette
python -m pixelart_creator input.jpg output.png --palette gameboy --dithering

# Apply filters
python -m pixelart_creator input.jpg output.png --filters contrast:1.5 sharpen:0.3 sepia
```

#### Getting Started Examples
```bash
# Most basic pixelation - just make it pixelated!
python -m pixelart_creator photo.jpg pixelated.png --scale-factor 10

# Quick Game Boy style conversion
python -m pixelart_creator photo.jpg gameboy.png --palette gameboy

# Make a small icon-sized version
python -m pixelart_creator logo.png icon.png --target-size 32 32

# Add some contrast to make it pop
python -m pixelart_creator photo.jpg enhanced.png --scale-factor 8 --filters contrast:1.3
```

#### Algorithm Showcase Examples
```bash
# Compare different algorithms with the same scale factor
python -m pixelart_creator input.jpg nearest.png --algorithm nearest --scale-factor 8
python -m pixelart_creator input.jpg bilinear.png --algorithm bilinear --scale-factor 8
python -m pixelart_creator input.jpg lanczos.png --algorithm lanczos --scale-factor 8
python -m pixelart_creator input.jpg edge.png --algorithm edge_preserving --scale-factor 8

# Super pixel for artistic clustering effect
python -m pixelart_creator landscape.jpg clustered.png --algorithm super_pixel --target-size 100 100

# Adaptive algorithm for content-aware pixelation
python -m pixelart_creator portrait.jpg adaptive.png --algorithm adaptive --scale-factor 12
```

#### Retro Gaming Styles
```bash
# Classic Game Boy green screen effect
python -m pixelart_creator photo.jpg gameboy.png \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering \
  --filters contrast:1.2 brightness:0.9

# NES-style with classic 8-bit colors
python -m pixelart_creator photo.jpg nes_style.png \
  --algorithm nearest \
  --palette nes \
  --scale-factor 6 \
  --filters contrast:1.1 saturation:1.2

# Commodore 64 nostalgic colors
python -m pixelart_creator photo.jpg c64_style.png \
  --algorithm bilinear \
  --palette c64 \
  --target-size 160 120 \
  --dithering

# Classic CGA 4-color style
python -m pixelart_creator photo.jpg cga_style.png \
  --algorithm nearest \
  --palette cga \
  --scale-factor 8 \
  --filters posterize:2
```

#### Modern Retro Effects
```bash
# CRT monitor simulation with scan lines
python -m pixelart_creator photo.jpg crt_effect.png \
  --scale-factor 10 \
  --filters scan_lines:3:0.4 vignette:0.5 chromatic_aberration:2

# VHS-style degraded look
python -m pixelart_creator photo.jpg vhs_style.png \
  --algorithm bilinear \
  --scale-factor 8 \
  --filters noise:0.1 blur:0.5 contrast:0.8 saturation:1.3

# Glitch art effect
python -m pixelart_creator photo.jpg glitch.png \
  --algorithm nearest \
  --scale-factor 12 \
  --filters chromatic_aberration:3 noise:0.2 solarize:128

# Retro terminal green screen
python -m pixelart_creator photo.jpg terminal.png \
  --palette monochrome \
  --algorithm edge_preserving \
  --filters contrast:1.5 brightness:0.7 scan_lines:2:0.3
```

#### Artistic and Creative Effects
```bash
# Sepia-toned vintage pixel art
python -m pixelart_creator photo.jpg vintage.png \
  --algorithm lanczos \
  --scale-factor 8 \
  --filters sepia:0.8 vignette:0.3 contrast:1.1

# High-contrast black and white
python -m pixelart_creator photo.jpg dramatic.png \
  --palette monochrome \
  --algorithm edge_preserving \
  --filters contrast:2.0 brightness:0.8

# Embossed 3D effect
python -m pixelart_creator photo.jpg embossed.png \
  --algorithm bilinear \
  --scale-factor 10 \
  --filters emboss grayscale contrast:1.5

# Posterized color reduction
python -m pixelart_creator photo.jpg poster.png \
  --algorithm nearest \
  --scale-factor 6 \
  --filters posterize:4 saturation:1.4 contrast:1.2

# Edge-enhanced line art style
python -m pixelart_creator photo.jpg line_art.png \
  --algorithm edge_preserving \
  --target-size 200 200 \
  --filters find_edges invert contrast:1.8
```

#### Size and Quality Examples
```bash
# Tiny 16x16 favicon (resized to 16x16)
python -m pixelart_creator logo.png favicon.png --target-size 16 16 --algorithm lanczos

# Medium game sprite (64x64 pixels)
python -m pixelart_creator character.png sprite.png --target-size 64 64 --palette nes

# Pixelated effect but keep original size
python -m pixelart_creator photo.jpg pixelated_effect.png --target-size 64 64 --keep-original-size

# Actually resize to small dimensions for true pixel art (default)
python -m pixelart_creator photo.jpg small_sprite.png --target-size 32 32

# Large pixelated artwork (scale factor approach)
python -m pixelart_creator photo.jpg large_pixel.png --scale-factor 4 --algorithm bilinear

# Ultra-pixelated for dramatic effect
python -m pixelart_creator photo.jpg ultra_pixel.png --scale-factor 20 --algorithm nearest

# High-quality downsampling for clean results
python -m pixelart_creator photo.jpg clean.png --target-size 128 128 --algorithm lanczos
```

#### Filter Combination Examples
```bash
# Warm vintage look
python -m pixelart_creator photo.jpg warm.png \
  --scale-factor 8 \
  --filters sepia:0.6 vignette:0.2 contrast:1.1 brightness:1.1

# Cool cyberpunk style
python -m pixelart_creator photo.jpg cyber.png \
  --scale-factor 10 \
  --filters saturation:0.7 contrast:1.4 chromatic_aberration:1 scan_lines:2:0.2

# Sharp and clean pixel art
python -m pixelart_creator photo.jpg sharp.png \
  --algorithm edge_preserving \
  --scale-factor 8 \
  --filters sharpen:0.5 contrast:1.2 edge_enhance

# Soft dreamy effect
python -m pixelart_creator photo.jpg dreamy.png \
  --algorithm bilinear \
  --scale-factor 12 \
  --filters blur:0.3 brightness:1.2 saturation:1.1 vignette:0.1

# Multiple artistic filters
python -m pixelart_creator photo.jpg artistic.png \
  --algorithm super_pixel \
  --target-size 80 80 \
  --filters posterize:6 saturation:1.3 contrast:1.1 sharpen:0.2
```

#### Advanced Technical Examples
```bash
# High-quality with dithering for smooth gradients
python -m pixelart_creator photo.jpg quality.png \
  --algorithm lanczos \
  --target-size 128 128 \
  --palette gameboy \
  --dithering \
  --filters contrast:1.1

# Super pixel clustering with specific target size
python -m pixelart_creator landscape.jpg clustered.png \
  --algorithm super_pixel \
  --target-size 120 80 \
  --filters saturation:1.2 contrast:1.1

# Adaptive algorithm with enhancement
python -m pixelart_creator portrait.jpg adaptive.png \
  --algorithm adaptive \
  --scale-factor 10 \
  --filters edge_enhance contrast:1.2 sharpen:0.3

# Edge preserving with color palette
python -m pixelart_creator photo.jpg preserved.png \
  --algorithm edge_preserving \
  --palette c64 \
  --scale-factor 8 \
  --dithering \
  --filters saturation:1.1
```

#### Batch Processing Examples
```bash
# Basic batch processing - convert all images in a folder
python -m pixelart_creator --batch \
  --input-dir ./photos \
  --output-dir ./pixel_art \
  --scale-factor 8

# Game Boy style batch conversion
python -m pixelart_creator --batch \
  --input-dir ./family_photos \
  --output-dir ./gameboy_gallery \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering \
  --scale-factor 10

# Process only JPEG files with specific settings
python -m pixelart_creator --batch \
  --input-dir ./images \
  --output-dir ./output \
  --pattern "*.jpg" \
  --algorithm super_pixel \
  --target-size 100 100 \
  --filters contrast:1.2 saturation:1.1

# Batch process with retro CRT effect
python -m pixelart_creator --batch \
  --input-dir ./screenshots \
  --output-dir ./retro_games \
  --scale-factor 8 \
  --filters scan_lines:3:0.4 vignette:0.3 \
  --algorithm nearest

# Create thumbnail gallery with consistent size
python -m pixelart_creator --batch \
  --input-dir ./portfolio \
  --output-dir ./pixel_thumbnails \
  --target-size 64 64 \
  --algorithm lanczos \
  --palette nes

# Batch artistic processing
python -m pixelart_creator --batch \
  --input-dir ./artwork \
  --output-dir ./pixel_art \
  --algorithm bilinear \
  --scale-factor 6 \
  --filters sepia:0.7 vignette:0.2 contrast:1.3

# Process specific file types with verbose output
python -m pixelart_creator --batch \
  --input-dir ./mixed_images \
  --output-dir ./processed \
  --pattern "*.{png,jpg,jpeg}" \
  --algorithm edge_preserving \
  --scale-factor 12 \
  --verbose
```

#### List Available Options
```bash
# List all algorithms
python -m pixelart_creator --list algorithms

# List all palettes
python -m pixelart_creator --list palettes

# List all filters
python -m pixelart_creator --list filters
```

#### Real-World Workflow Examples

**Game Development Workflow:**
```bash
# Convert character portraits to consistent game style
python -m pixelart_creator --batch \
  --input-dir ./character_art \
  --output-dir ./game_sprites \
  --target-size 64 64 \
  --algorithm edge_preserving \
  --palette nes \
  --filters contrast:1.2

# Create different resolution assets (now defaults to resizing)
python -m pixelart_creator hero.png hero_32.png --target-size 32 32 --algorithm lanczos
python -m pixelart_creator hero.png hero_64.png --target-size 64 64 --algorithm lanczos
python -m pixelart_creator hero.png hero_128.png --target-size 128 128 --algorithm lanczos
```

**Social Media Content Creation:**
```bash
# Instagram retro filter effect (keep original size for social media)
python -m pixelart_creator selfie.jpg insta_retro.png \
  --scale-factor 8 \
  --filters sepia:0.6 vignette:0.3 contrast:1.2 scan_lines:4:0.2

# TikTok-style glitch effect (keep original size)
python -m pixelart_creator video_frame.png tiktok_glitch.png \
  --algorithm nearest \
  --scale-factor 10 \
  --filters chromatic_aberration:3 noise:0.15 solarize:100

# Profile picture pixel art (resize to standard profile size)
python -m pixelart_creator profile.jpg pixel_avatar.png \
  --target-size 128 128 \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering
```

**Art and Design Projects:**
```bash
# Convert photos to pixel art paintings
python -m pixelart_creator landscape.jpg pixel_painting.png \
  --algorithm super_pixel \
  --target-size 200 150 \
  --filters posterize:8 saturation:1.3 contrast:1.1

# Create consistent art style for comic/story
python -m pixelart_creator --batch \
  --input-dir ./comic_panels \
  --output-dir ./pixel_comic \
  --algorithm edge_preserving \
  --scale-factor 6 \
  --filters contrast:1.3 edge_enhance

# Vintage poster effect
python -m pixelart_creator poster.jpg vintage_pixel.png \
  --algorithm bilinear \
  --scale-factor 8 \
  --filters sepia:0.8 posterize:6 vignette:0.4 contrast:1.4
```

**Web Development Assets:**
```bash
# Create favicon and web icons (resize to exact dimensions)
python -m pixelart_creator logo.png favicon-16.png --target-size 16 16 --algorithm lanczos
python -m pixelart_creator logo.png icon-32.png --target-size 32 32 --algorithm lanczos
python -m pixelart_creator logo.png icon-64.png --target-size 64 64 --algorithm lanczos

# Retro website banner (resize to banner dimensions)
python -m pixelart_creator banner.jpg retro_banner.png \
  --target-size 800 200 \
  --algorithm bilinear \
  --palette c64 \
  --filters scan_lines:2:0.1 vignette:0.2
```

**Digital Art Experiments:**
```bash
# Abstract pixel art from photos
python -m pixelart_creator abstract.jpg pixel_abstract.png \
  --algorithm super_pixel \
  --scale-factor 15 \
  --filters posterize:4 saturation:2.0 solarize:120

# Minimalist style with limited colors
python -m pixelart_creator portrait.jpg minimalist.png \
  --algorithm adaptive \
  --palette monochrome \
  --dithering \
  --filters contrast:1.8 edge_enhance

# Cyberpunk aesthetic
python -m pixelart_creator cityscape.jpg cyberpunk.png \
  --algorithm nearest \
  --scale-factor 8 \
  --filters saturation:0.6 contrast:1.5 chromatic_aberration:2 scan_lines:3:0.3
```

### Graphical User Interface

Launch the GUI:
```bash
python -m pixelart_creator --gui
```

The GUI provides:
- File browser for input/output selection
- Real-time parameter adjustment
- Live preview window
- One-click processing and saving

## Available Algorithms

1. **Nearest Neighbor** - Simple downsampling with crisp edges
2. **Bilinear** - Smoother downsampling with bilinear interpolation
3. **Lanczos** - High-quality downsampling with Lanczos filter
4. **Edge Preserving** - Maintains important image boundaries
5. **Super Pixel** - Groups similar pixels using SLIC segmentation
6. **Adaptive** - Variable pixelation based on local image complexity

## Available Palettes

### Predefined Palettes
- **Game Boy** (4 colors) - Classic green monochrome
- **NES** (14 colors) - Nintendo Entertainment System palette
- **C64** (16 colors) - Commodore 64 color scheme
- **CGA** (16 colors) - Color Graphics Adapter palette
- **Monochrome** (4 colors) - Black and white gradient

### Custom Palettes
Create JSON files with the format:
```json
{
  "name": "My Palette",
  "colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
}
```

## Available Filters

### Basic Adjustments
- `contrast:factor` - Adjust image contrast
- `brightness:factor` - Adjust image brightness
- `saturation:factor` - Adjust color saturation

### Artistic Effects
- `blur:radius` - Gaussian blur
- `sharpen:factor` - Image sharpening
- `emboss` - Embossed 3D effect
- `sepia:intensity` - Sepia tone effect
- `posterize:bits` - Reduce color levels
- `solarize:threshold` - Solarization effect

### Retro Effects
- `scan_lines:height:opacity` - CRT-style scan lines
- `chromatic_aberration:offset` - Lens chromatic aberration
- `vignette:intensity` - Dark edge vignetting
- `noise:intensity` - Random noise

### Utility Filters
- `grayscale` - Convert to grayscale
- `invert` - Invert colors
- `edge_enhance` - Enhance edges
- `find_edges` - Edge detection

## Project Structure

```
pixelart_creator/
├── __init__.py              # Package initialization
├── __main__.py              # Main entry point
├── core/                    # Core functionality
│   ├── __init__.py
│   └── image_processor.py   # Main image processing class
├── algorithms/              # Pixelation algorithms
│   ├── __init__.py
│   ├── base.py             # Algorithm base classes
│   ├── nearest_neighbor.py # Simple algorithms
│   ├── advanced.py         # Advanced algorithms
│   └── manager.py          # Algorithm management
├── palettes/               # Color palette management
│   ├── __init__.py
│   └── color_palette.py    # Palette classes and presets
├── filters/                # Image filters and effects
│   ├── __init__.py
│   └── image_filters.py    # Filter implementations
├── gui/                    # Graphical user interface
│   ├── __init__.py
│   └── main_window.py      # Main GUI window
├── cli/                    # Command line interface
│   ├── __init__.py
│   └── interface.py        # CLI implementation
└── utils/                  # Utility modules
    ├── __init__.py
    └── exceptions.py       # Custom exceptions
```

## Examples and Tutorials

### Creating Test Images
```bash
python create_test_image.py
```
This creates a sample image in `test_images/test_image.png` for testing all the examples below.

### Quick Start Tutorial

**Step 1: Basic Pixelation**
```bash
# Start with simple pixelation to understand the basics
python -m pixelart_creator test_images/test_image.png tutorial_step1.png --scale-factor 8
```

**Step 2: Try Different Algorithms**
```bash
# Compare how different algorithms affect the same image
python -m pixelart_creator test_images/test_image.png nearest.png --algorithm nearest --scale-factor 8
python -m pixelart_creator test_images/test_image.png edge.png --algorithm edge_preserving --scale-factor 8
python -m pixelart_creator test_images/test_image.png super.png --algorithm super_pixel --target-size 80 80
```

**Step 3: Add Color Palettes**
```bash
# Experience classic gaming color schemes
python -m pixelart_creator test_images/test_image.png gameboy.png --palette gameboy --scale-factor 8
python -m pixelart_creator test_images/test_image.png nes.png --palette nes --scale-factor 8
python -m pixelart_creator test_images/test_image.png c64.png --palette c64 --scale-factor 8
```

**Step 4: Apply Filters for Effects**
```bash
# Add retro effects and enhancements
python -m pixelart_creator test_images/test_image.png retro.png --scale-factor 8 --filters scan_lines:3:0.4 vignette:0.3
python -m pixelart_creator test_images/test_image.png enhanced.png --scale-factor 8 --filters contrast:1.3 sharpen:0.2
python -m pixelart_creator test_images/test_image.png artistic.png --scale-factor 8 --filters sepia:0.7 emboss
```

**Step 5: Combine Everything**
```bash
# Create a masterpiece combining algorithm, palette, and filters
python -m pixelart_creator test_images/test_image.png masterpiece.png \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering \
  --scale-factor 10 \
  --filters contrast:1.2 vignette:0.2
```

### Complete Example Workflows

**Creating Game Boy Style Art:**
```bash
# Perfect Game Boy aesthetic with all the right settings
python -m pixelart_creator your_photo.jpg gameboy_perfect.png \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering \
  --scale-factor 8 \
  --filters contrast:1.2 brightness:0.9
```

**Modern Retro CRT Effect:**
```bash
# Simulate old CRT monitor with scan lines and color bleeding
python -m pixelart_creator your_photo.jpg crt_monitor.png \
  --algorithm bilinear \
  --scale-factor 12 \
  --filters scan_lines:3:0.4 chromatic_aberration:2 vignette:0.5 contrast:1.1
```

**High-Quality Pixel Art:**
```bash
# Best quality settings for detailed pixel art
python -m pixelart_creator your_photo.jpg high_quality.png \
  --algorithm lanczos \
  --target-size 160 160 \
  --filters edge_enhance sharpen:0.3 contrast:1.1
```

**Artistic Black and White:**
```bash
# Dramatic monochrome pixel art
python -m pixelart_creator your_photo.jpg dramatic_bw.png \
  --algorithm edge_preserving \
  --palette monochrome \
  --dithering \
  --scale-factor 10 \
  --filters contrast:2.0 brightness:0.8
```

### Batch Processing Examples

**Family Photo Gallery:**
```bash
# Convert all family photos to consistent Game Boy style
python -m pixelart_creator --batch \
  --input-dir ./family_photos \
  --output-dir ./gameboy_family \
  --algorithm edge_preserving \
  --palette gameboy \
  --dithering \
  --scale-factor 8 \
  --verbose
```

**Social Media Content Batch:**
```bash
# Process multiple images for social media with retro filter
python -m pixelart_creator --batch \
  --input-dir ./social_content \
  --output-dir ./retro_content \
  --scale-factor 10 \
  --filters sepia:0.6 vignette:0.3 scan_lines:4:0.2 \
  --pattern "*.{jpg,png}"
```

### Troubleshooting Examples

**If output is too pixelated:**
```bash
# Use smaller scale factor or larger target size
python -m pixelart_creator input.jpg less_pixelated.png --scale-factor 4
# or
python -m pixelart_creator input.jpg less_pixelated.png --target-size 200 200
```

**If you want the original image size preserved:**
```bash
# Use --keep-original-size for pixelated effects without resizing
python -m pixelart_creator input.jpg pixelated_effect.png --target-size 64 64 --keep-original-size
```

**If colors look wrong:**
```bash
# Try different algorithm or add contrast
python -m pixelart_creator input.jpg better_colors.png --algorithm edge_preserving --filters contrast:1.2
```

**If image looks too soft:**
```bash
# Use nearest neighbor algorithm for crisp pixels
python -m pixelart_creator input.jpg crisp.png --algorithm nearest --scale-factor 8
```

**If you want smoother gradients:**
```bash
# Use dithering with a color palette
python -m pixelart_creator input.jpg smooth.png --palette gameboy --dithering --algorithm bilinear
```

## Architecture

### Core Components

1. **ImageProcessor** - Central class for image manipulation
2. **AlgorithmManager** - Handles algorithm registration and instantiation
3. **ColorPalette** - Manages color palettes and application
4. **ImageFilters** - Provides image effects and filters
5. **CLI Interface** - Command-line argument parsing and processing
6. **GUI Interface** - Tkinter-based graphical interface

### Design Patterns

- **Strategy Pattern** - Algorithm selection and execution
- **Factory Pattern** - Algorithm and palette creation
- **Observer Pattern** - GUI event handling
- **Command Pattern** - CLI command processing

### Error Handling

Custom exception hierarchy for different error types:
- `PixelArtError` - Base exception
- `UnsupportedFormatError` - Invalid file formats
- `InvalidAlgorithmError` - Algorithm errors
- `InvalidPaletteError` - Palette errors
- `ProcessingError` - Image processing failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Adding New Algorithms

1. Create a new class inheriting from `PixelationAlgorithm`
2. Implement the `apply()` method
3. Register the algorithm in `manager.py`

### Adding New Filters

1. Create a filter function with the `@ImageFilters.register_filter` decorator
2. Add parameter validation and error handling
3. Update the filter documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by classic video game aesthetics
- Uses various image processing libraries
- Built with extensibility and usability in mind
