"""
GUI application for the PixelArt Creator.

This module provides a simple Tkinter-based GUI for converting images to pixel art.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import threading
from typing import Optional

from ..core.image_processor import ImageProcessor
from ..algorithms.manager import AlgorithmManager
from ..palettes.color_palette import PredefinedPalettes
from ..utils.exceptions import PixelArtError


class PixelArtGUI:
    """Main GUI application for PixelArt Creator."""
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("PixelArt Creator")
        self.root.geometry("800x600")
        
        # Application state
        self.processor = ImageProcessor()
        self.original_image = None
        self.processed_image = None
        self.preview_image = None
        
        # Create UI
        self.create_widgets()
        self.setup_layout()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # File selection frame
        self.file_frame = ttk.LabelFrame(self.main_frame, text="File Selection", padding="5")
        
        self.input_label = ttk.Label(self.file_frame, text="Input Image:")
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(self.file_frame, textvariable=self.input_var, width=50)
        self.input_button = ttk.Button(self.file_frame, text="Browse", 
                                      command=self.browse_input_file)
        
        self.output_label = ttk.Label(self.file_frame, text="Output Image:")
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.file_frame, textvariable=self.output_var, width=50)
        self.output_button = ttk.Button(self.file_frame, text="Browse", 
                                       command=self.browse_output_file)
        
        # Settings frame
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="Pixelation Settings", padding="5")
        
        # Scale factor
        self.scale_label = ttk.Label(self.settings_frame, text="Scale Factor:")
        self.scale_var = tk.DoubleVar(value=8.0)
        self.scale_scale = ttk.Scale(self.settings_frame, from_=1, to=32, 
                                   variable=self.scale_var, orient="horizontal")
        self.scale_value_label = ttk.Label(self.settings_frame, text="8.0")
        self.scale_var.trace('w', self.update_scale_label)
        
        # Algorithm selection
        self.algorithm_label = ttk.Label(self.settings_frame, text="Algorithm:")
        self.algorithm_var = tk.StringVar(value="nearest")
        algorithms = AlgorithmManager.get_available_algorithms()
        self.algorithm_combo = ttk.Combobox(self.settings_frame, textvariable=self.algorithm_var,
                                          values=algorithms, state="readonly")
        
        # Palette selection
        self.palette_label = ttk.Label(self.settings_frame, text="Color Palette:")
        self.palette_var = tk.StringVar(value="none")
        palettes = ["none"] + list(PredefinedPalettes.get_all_palettes().keys())
        self.palette_combo = ttk.Combobox(self.settings_frame, textvariable=self.palette_var,
                                        values=palettes, state="readonly")
        
        # Dithering option
        self.dithering_var = tk.BooleanVar()
        self.dithering_check = ttk.Checkbutton(self.settings_frame, text="Apply Dithering",
                                             variable=self.dithering_var)
        
        # Preview frame
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="Preview", padding="5")
        
        # Canvas for image preview
        self.canvas = tk.Canvas(self.preview_frame, width=400, height=300, bg="white")
        self.canvas_h_scroll = ttk.Scrollbar(self.preview_frame, orient="horizontal", 
                                           command=self.canvas.xview)
        self.canvas_v_scroll = ttk.Scrollbar(self.preview_frame, orient="vertical", 
                                           command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.canvas_h_scroll.set,
                            yscrollcommand=self.canvas_v_scroll.set)
        
        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        
        self.load_button = ttk.Button(self.button_frame, text="Load Image", 
                                    command=self.load_image)
        self.preview_button = ttk.Button(self.button_frame, text="Preview", 
                                       command=self.preview_pixelation)
        self.process_button = ttk.Button(self.button_frame, text="Process & Save", 
                                       command=self.process_and_save)
        self.reset_button = ttk.Button(self.button_frame, text="Reset", 
                                     command=self.reset_image)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, 
                                  relief="sunken", anchor="w")
        
    def setup_layout(self):
        """Set up the widget layout."""
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # File selection
        self.file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.file_frame.columnconfigure(1, weight=1)
        
        self.input_label.grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.input_button.grid(row=0, column=2, sticky="e")
        
        self.output_label.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=(5, 0))
        self.output_button.grid(row=1, column=2, sticky="e", pady=(5, 0))
        
        # Settings
        self.settings_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10), padx=(0, 10))
        
        self.scale_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.scale_scale.grid(row=0, column=1, sticky="ew", padx=(5, 5), pady=(0, 5))
        self.scale_value_label.grid(row=0, column=2, sticky="w", pady=(0, 5))
        
        self.algorithm_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.algorithm_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        self.palette_label.grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.palette_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        self.dithering_check.grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))
        
        self.settings_frame.columnconfigure(1, weight=1)
        
        # Preview
        self.preview_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas_v_scroll.grid(row=0, column=1, sticky="ns")
        
        # Buttons
        self.button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.load_button.grid(row=0, column=0, padx=(0, 5))
        self.preview_button.grid(row=0, column=1, padx=(0, 5))
        self.process_button.grid(row=0, column=2, padx=(0, 5))
        self.reset_button.grid(row=0, column=3)
        
        # Status bar
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")
        
        self.main_frame.rowconfigure(1, weight=1)
        
    def update_scale_label(self, *args):
        """Update the scale factor label."""
        value = self.scale_var.get()
        self.scale_value_label.config(text=f"{value:.1f}")
        
    def browse_input_file(self):
        """Browse for input file."""
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_var.set(filename)
            # Auto-generate output filename
            input_path = Path(filename)
            output_path = input_path.parent / f"{input_path.stem}_pixelated{input_path.suffix}"
            self.output_var.set(str(output_path))
            
    def browse_output_file(self):
        """Browse for output file."""
        filename = filedialog.asksaveasfilename(
            title="Save Pixelated Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.output_var.set(filename)
            
    def load_image(self):
        """Load the input image."""
        input_path = self.input_var.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input image file.")
            return
            
        try:
            self.status_var.set("Loading image...")
            self.root.update()
            
            self.processor.load_image(input_path)
            self.original_image = self.processor.current_image.copy()
            
            # Display original image in canvas
            self.display_image(self.original_image)
            
            self.status_var.set(f"Loaded image: {self.original_image.size}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.status_var.set("Ready")
            
    def preview_pixelation(self):
        """Preview the pixelation without saving."""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
            
        try:
            self.status_var.set("Generating preview...")
            self.root.update()
            
            # Get settings
            algorithm = AlgorithmManager.get_algorithm(self.algorithm_var.get())
            if not algorithm:
                messagebox.showerror("Error", f"Algorithm '{self.algorithm_var.get()}' not found.")
                return
                
            scale_factor = self.scale_var.get()
            original_size = self.original_image.size
            target_size = (
                int(original_size[0] / scale_factor),
                int(original_size[1] / scale_factor)
            )
            
            # Apply pixelation
            temp_processor = ImageProcessor()
            temp_processor.current_image = self.original_image.copy()
            temp_processor.pixelate(algorithm, target_size)
            
            # Apply palette if selected
            palette_name = self.palette_var.get()
            if palette_name != "none":
                palettes = PredefinedPalettes.get_all_palettes()
                if palette_name in palettes:
                    palette = palettes[palette_name]
                    temp_processor.apply_palette(palette)
            
            self.preview_image = temp_processor.current_image
            self.display_image(self.preview_image)
            
            self.status_var.set("Preview generated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate preview: {e}")
            self.status_var.set("Ready")
            
    def process_and_save(self):
        """Process the image and save to file."""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
            
        output_path = self.output_var.get()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return
            
        def process_thread():
            try:
                self.status_var.set("Processing...")
                
                # Get settings
                algorithm = AlgorithmManager.get_algorithm(self.algorithm_var.get())
                scale_factor = self.scale_var.get()
                original_size = self.original_image.size
                target_size = (
                    int(original_size[0] / scale_factor),
                    int(original_size[1] / scale_factor)
                )
                
                # Process image
                temp_processor = ImageProcessor()
                temp_processor.current_image = self.original_image.copy()
                temp_processor.pixelate(algorithm, target_size)
                
                # Apply palette if selected
                palette_name = self.palette_var.get()
                if palette_name != "none":
                    palettes = PredefinedPalettes.get_all_palettes()
                    if palette_name in palettes:
                        palette = palettes[palette_name]
                        temp_processor.apply_palette(palette)
                
                # Save image
                temp_processor.save_image(output_path)
                
                self.processed_image = temp_processor.current_image
                self.display_image(self.processed_image)
                
                self.root.after(0, lambda: self.status_var.set(f"Saved to: {output_path}"))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Image saved to: {output_path}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process image: {e}"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        # Run processing in separate thread to avoid blocking UI
        threading.Thread(target=process_thread, daemon=True).start()
        
    def reset_image(self):
        """Reset to original image."""
        if self.original_image:
            self.display_image(self.original_image)
            self.processor.current_image = self.original_image.copy()
            self.status_var.set("Reset to original image")
        
    def display_image(self, image: Image.Image):
        """Display an image in the canvas."""
        try:
            # Resize image for display if too large
            display_image = image.copy()
            max_size = (400, 300)
            
            if display_image.size[0] > max_size[0] or display_image.size[1] > max_size[1]:
                display_image.thumbnail(max_size, Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for GUI."""
    try:
        app = PixelArtGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
