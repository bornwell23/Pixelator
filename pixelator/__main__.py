"""
Main entry point for the PixelArt Creator application.
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main entry point that determines whether to run CLI or GUI."""
    # Simple argument parsing to determine mode
    parser = argparse.ArgumentParser(
        description='PixelArt Creator - Convert images to pixel art',
        add_help=False
    )
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')
    parser.add_argument('--help', action='store_true', help='Show help')
    
    # Parse known args to check for GUI flag
    known_args, remaining = parser.parse_known_args()
    
    if known_args.gui:
        # Launch GUI
        try:
            from .gui.main_window import PixelArtGUI
            app = PixelArtGUI()
            app.run()
        except ImportError:
            print("Error: GUI dependencies not available")
            print("Please install tkinter for GUI support")
            sys.exit(1)
        except Exception as e:
            print(f"Error launching GUI: {e}")
            sys.exit(1)
    else:
        # Use CLI
        try:
            from .cli.interface import main as cli_main
            cli_main()
        except ImportError as e:
            print(f"Import error: {e}")
            print("Make sure all dependencies are installed:")
            print("pip install pillow numpy click opencv-python scikit-image scikit-learn")
            sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
