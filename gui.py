"""
GUI for Cephalometric Landmark DetecPrediction

Features:
- Load and analyze cephalometric X-ray images
- Compare predictions with ground truth annotations
- Calculate MRE and SDR metrics
- Support argmax and soft_argmax coordinate extraction

Usage:
    python gui.py
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
import os
import sys
import json
import glob
import csv

sys.path.insert(0, 'src')
from src.models.hrnet import HRNet
from src.utils.metrics import extract_coordinates_from_heatmaps, soft_argmax_2d
from src.config.config import TrainConfig

# Landmark names in JSON file order (as used by dataset.py during training)
# This is the ACTUAL order from the annotation files
LANDMARK_NAMES = [
    "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion",
    "Orbitale", "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus",
    "Sella", "Articulare", "Condylion", "Gnathion", "Gonion",
    "Porion", "Lower 2nd PM Cusp Tip", "Lower Incisor Tip", "Lower Molar Cusp Tip",
    "Upper 2nd PM Cusp Tip", "Upper Incisor Apex", "Upper Incisor Tip", 
    "Upper Molar Cusp Tip", "Lower Incisor Apex", "Labrale inferius",
    "Labrale superius", "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale"
]

# Short names for display (matching JSON order)
LANDMARK_SHORT = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Prn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "L5", "L1", "L6",
    "U5", "U1A", "U1", "U6", "L1A", "Li", "Ls", "N'", "Pog'", "Sn"
]


class CephalometricGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cephalometric Landmark Prediction - With Ground Truth")
        self.root.geometry("1800x950")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.landmarks = None
        self.confidences = None
        self.gt_landmarks = None
        self.model = None
        self.config = TrainConfig()
        self.show_predictions = tk.BooleanVar(value=True)
        self.show_ground_truth = tk.BooleanVar(value=True)
        self.show_lines = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.coord_method = tk.StringVar(value=getattr(self.config, 'COORD_EXTRACTION_METHOD', 'argmax'))
        self.model_path = None
        self.model_checkpoint_meta = None
        self.current_pixel_spacing = self.config.DEFAULT_PIXEL_SPACING
        self.pixel_spacing_by_id = {}
        
        # Dataset path
        self.dataset_path = r"C:\Users\clyde\Special Problem\AI Cephalometric Model_Xenodent\dataset"

        self._load_pixel_spacing_map()
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
        
        # Legend window (hidden by default)
        self.legend_window = None

    def _load_pixel_spacing_map(self):
        mapping_path = os.path.join(self.dataset_path, "cephalogram_machine_mappings.csv")
        if not os.path.exists(mapping_path):
            return

        try:
            with open(mapping_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ceph_id = row.get("cephalogram_id")
                    px = row.get("pixel_size")
                    if ceph_id and px:
                        try:
                            self.pixel_spacing_by_id[ceph_id] = float(px)
                        except Exception:
                            continue
        except Exception:
            return

    def _set_current_pixel_spacing(self, image_id: str | None):
        # Simplified: use default pixel spacing for all images
        self.current_pixel_spacing = self.config.DEFAULT_PIXEL_SPACING
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = HRNet(
                num_landmarks=self.config.NUM_LANDMARKS,
                use_multitask=self.config.USE_MULTITASK
            )
            
            checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_path = checkpoint_path
                if isinstance(checkpoint, dict):
                    self.model_checkpoint_meta = {
                        'epoch': checkpoint.get('epoch', None),
                        'mre': checkpoint.get('mre', None),
                        'best_mre': checkpoint.get('best_mre', None),
                        'best_epoch': checkpoint.get('best_epoch', None),
                    }
                print(f"Loaded model from {checkpoint_path}")
                ckpt_epoch = checkpoint.get('epoch', None)
                try:
                    if ckpt_epoch is None:
                        print("  Epoch: N/A")
                    else:
                        # train.py stores 0-based epoch in checkpoint; logs print epoch+1
                        print(f"  Epoch: {int(ckpt_epoch) + 1}")
                except Exception:
                    print(f"  Epoch: {ckpt_epoch}")
                best_mre = checkpoint.get('best_mre', None)
                if best_mre is not None:
                    print(f"  Best MRE: {best_mre:.2f} mm")
            else:
                messagebox.showwarning("Warning", f"No checkpoint found at {checkpoint_path}")
            
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_model_from_path(self, checkpoint_path: str):
        """Load a model checkpoint selected by the user."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Support both formats:
            # 1) Our training checkpoints: {'model_state_dict': ..., 'epoch': ..., ...}
            # 2) Raw state_dict saved directly
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

            if self.model is None:
                self.model = HRNet(
                    num_landmarks=self.config.NUM_LANDMARKS,
                    use_multitask=self.config.USE_MULTITASK
                )

            self.model.load_state_dict(state_dict)
            self.model.eval()

            if not hasattr(self, 'device'):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)

            self.model_path = checkpoint_path

            epoch = checkpoint.get('epoch', None) if isinstance(checkpoint, dict) else None
            best_mre = checkpoint.get('best_mre', None) if isinstance(checkpoint, dict) else None
            if isinstance(checkpoint, dict):
                self.model_checkpoint_meta = {
                    'epoch': checkpoint.get('epoch', None),
                    'mre': checkpoint.get('mre', None),
                    'best_mre': checkpoint.get('best_mre', None),
                    'best_epoch': checkpoint.get('best_epoch', None),
                }
            else:
                self.model_checkpoint_meta = None

            msg = f"Loaded model: {os.path.basename(checkpoint_path)}"
            if epoch is not None:
                msg += f" (epoch {epoch})"
            if best_mre is not None:
                msg += f" | best MRE {best_mre:.2f}mm"

            if hasattr(self, 'status_label'):
                self.status_label.config(text=msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load selected model:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def select_model_file(self):
        """Open a file picker to select a .pth checkpoint file."""
        initial_dir = self.config.CHECKPOINT_DIR if os.path.isdir(self.config.CHECKPOINT_DIR) else os.getcwd()
        filetypes = [("PyTorch checkpoint", "*.pth"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select model checkpoint (.pth)", initialdir=initial_dir, filetypes=filetypes)
        if path:
            self.load_model_from_path(path)
            
    def create_ui(self):
        """Create the user interface"""
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#3c3c3c', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            left_panel, 
            text="Cephalometric\nLandmark Prediction",
            font=('Segoe UI', 14, 'bold'),
            bg='#3c3c3c',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Upload buttons frame
        upload_frame = tk.Frame(left_panel, bg='#3c3c3c')
        upload_frame.pack(pady=5, fill=tk.X, padx=10)
        
        # Upload any image
        self.upload_btn = tk.Button(
            upload_frame,
            text="📁 Upload Image",
            font=('Segoe UI', 10),
            bg='#0078d4',
            fg='white',
            cursor='hand2',
            command=self.upload_image,
            width=15
        )
        self.upload_btn.pack(side=tk.LEFT, padx=2)
        
        # Load from test set
        self.test_btn = tk.Button(
            upload_frame,
            text="📂 From Test Set",
            font=('Segoe UI', 10),
            bg='#6b4c9a',
            fg='white',
            cursor='hand2',
            command=self.load_from_test_set,
            width=15
        )
        self.test_btn.pack(side=tk.RIGHT, padx=2)

        # Load model button
        self.model_btn = tk.Button(
            left_panel,
            text="🧠 Load Model (.pth)",
            font=('Segoe UI', 10),
            bg='#444444',
            fg='white',
            cursor='hand2',
            command=self.select_model_file,
            width=25
        )
        self.model_btn.pack(pady=6)
        
        # Display options
        options_frame = tk.LabelFrame(left_panel, text="Display Options", 
                                       bg='#3c3c3c', fg='white', font=('Segoe UI', 10))
        options_frame.pack(pady=10, fill=tk.X, padx=10)
        
        tk.Checkbutton(options_frame, text="Show Predictions (●)", 
                      variable=self.show_predictions, bg='#3c3c3c', fg='#00ff00',
                      selectcolor='#2b2b2b', activebackground='#3c3c3c',
                      command=self.refresh_display).pack(anchor=tk.W, padx=5)
        
        tk.Checkbutton(options_frame, text="Show Ground Truth (■)", 
                      variable=self.show_ground_truth, bg='#3c3c3c', fg='#ff6600',
                      selectcolor='#2b2b2b', activebackground='#3c3c3c',
                      command=self.refresh_display).pack(anchor=tk.W, padx=5)
        
        tk.Checkbutton(options_frame, text="Show Error Lines", 
                      variable=self.show_lines, bg='#3c3c3c', fg='#ffffff',
                      selectcolor='#2b2b2b', activebackground='#3c3c3c',
                      command=self.refresh_display).pack(anchor=tk.W, padx=5)
        
        tk.Checkbutton(options_frame, text="Show Labels", 
                      variable=self.show_labels, bg='#3c3c3c', fg='#00ffff',
                      selectcolor='#2b2b2b', activebackground='#3c3c3c',
                      command=self.refresh_display).pack(anchor=tk.W, padx=5)

        method_frame = tk.Frame(options_frame, bg='#3c3c3c')
        method_frame.pack(fill=tk.X, padx=5, pady=(4, 2))
        tk.Label(method_frame, text="Decode:", bg='#3c3c3c', fg='white', font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.coord_method,
            values=['argmax', 'soft_argmax'],
            state='readonly',
            width=12,
        )
        self.method_combo.pack(side=tk.RIGHT)
        
        # Confidence threshold removed: current "confidence" is not calibrated and was misleading.
        
        # Analyze button
        self.analyze_btn = tk.Button(
            left_panel,
            text="🔍 Analyze",
            font=('Segoe UI', 11, 'bold'),
            bg='#107c10',
            fg='white',
            cursor='hand2',
            command=self.analyze_image,
            width=25,
            height=2,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(pady=10)

        # Metrics display
        metrics_frame = tk.LabelFrame(left_panel, text="Comparison Metrics", 
                                       bg='#3c3c3c', fg='white', font=('Segoe UI', 10))
        metrics_frame.pack(pady=5, fill=tk.X, padx=10)
        
        self.mre_label = tk.Label(metrics_frame, text="MRE: --", font=('Segoe UI', 11, 'bold'),
                                  bg='#3c3c3c', fg='#00ffff')
        self.mre_label.pack(anchor=tk.W, padx=5)
        
        self.sdr_label = tk.Label(metrics_frame, text="SDR@2mm: --", font=('Segoe UI', 10),
                                  bg='#3c3c3c', fg='#aaaaaa')
        self.sdr_label.pack(anchor=tk.W, padx=5)
        
        self.gt_status = tk.Label(metrics_frame, text="GT: Not loaded", font=('Segoe UI', 9),
                                  bg='#3c3c3c', fg='#666666')
        self.gt_status.pack(anchor=tk.W, padx=5, pady=2)
        
        # Status
        self.status_label = tk.Label(left_panel, text="Ready", font=('Segoe UI', 9),
                                     bg='#3c3c3c', fg='#aaaaaa')
        self.status_label.pack(pady=5)

        if self.model_path is not None:
            self.status_label.config(text=f"Model: {os.path.basename(self.model_path)}")
        
        # Landmark list with scrollbars
        list_frame = tk.Frame(left_panel, bg='#3c3c3c')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(list_frame, text="Landmarks (Pred vs GT):", font=('Segoe UI', 10, 'bold'),
                bg='#3c3c3c', fg='white').pack(anchor=tk.W)
        
        # Create frame for listbox and scrollbars
        listbox_frame = tk.Frame(list_frame, bg='#3c3c3c')
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(listbox_frame, orient="horizontal")
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.landmark_listbox = tk.Listbox(
            listbox_frame, font=('Consolas', 8), bg='#1e1e1e', fg='#cccccc',
            selectbackground='#0078d4', yscrollcommand=v_scrollbar.set, 
            xscrollcommand=h_scrollbar.set, height=18
        )
        self.landmark_listbox.pack(fill=tk.BOTH, expand=True)
        v_scrollbar.config(command=self.landmark_listbox.yview)
        h_scrollbar.config(command=self.landmark_listbox.xview)
        
        # Legend
        legend_frame = tk.Frame(left_panel, bg='#3c3c3c')
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(legend_frame, text="● Prediction  ■ Ground Truth", 
                font=('Segoe UI', 9), bg='#3c3c3c', fg='#888888').pack()
        
        # Right panel - Image and Legend
        right_panel = tk.Frame(main_frame, bg='#1e1e1e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Legend panel on right side
        legend_frame = tk.LabelFrame(right_panel, text="Landmark Legend", 
                                     bg='#3c3c3c', fg='white', font=('Segoe UI', 10))
        legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        legend_frame.configure(width=280)  # Set fixed width for legend panel
        legend_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Simple frame for legend (no scrollbars)
        legend_content = tk.Frame(legend_frame, bg='#3c3c3c')
        legend_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Headers
        header_frame = tk.Frame(legend_content, bg='#3c3c3c')
        header_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(header_frame, text="No.", font=('Segoe UI', 10, 'bold'), 
                bg='#3c3c3c', fg='white', width=3).pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Abbr.", font=('Segoe UI', 10, 'bold'), 
                bg='#3c3c3c', fg='white', width=5).pack(side=tk.LEFT, padx=1)
        tk.Label(header_frame, text="Name", font=('Segoe UI', 10, 'bold'), 
                bg='#3c3c3c', fg='white').pack(side=tk.LEFT, padx=1)
        
        # Separator
        separator = tk.Frame(legend_content, height=1, bg='#555555')
        separator.pack(fill=tk.X, pady=2)
        
        # Landmark entries (all 29 landmarks)
        for i, (short_name, full_name) in enumerate(zip(LANDMARK_SHORT, LANDMARK_NAMES)):
            entry_frame = tk.Frame(legend_content, bg='#3c3c3c')
            entry_frame.pack(fill=tk.X, pady=1)
            
            # Alternating row colors
            bg_color = '#444444' if i % 2 == 0 else '#3c3c3c'
            entry_frame.configure(bg=bg_color)
            
            tk.Label(entry_frame, text=f"{i+1:2d}", font=('Segoe UI', 8), 
                    bg=bg_color, fg='#aaaaaa', width=3).pack(side=tk.LEFT, padx=1)
            tk.Label(entry_frame, text=short_name, font=('Segoe UI', 8, 'bold'), 
                    bg=bg_color, fg='#00ffff', width=5).pack(side=tk.LEFT, padx=1)
            tk.Label(entry_frame, text=full_name[:25] + "..." if len(full_name) > 25 else full_name, 
                    font=('Segoe UI', 8), bg=bg_color, fg='white').pack(side=tk.LEFT, padx=1)
        
        # Canvas for image
        self.canvas = tk.Canvas(right_panel, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.create_text(500, 400, text="Upload an image or select from test set",
                               font=('Segoe UI', 14), fill='#666666', tags='placeholder')
        
    def update_conf_label(self, value=None):
        # Kept for backwards compatibility if referenced elsewhere.
        return
        
    def upload_image(self):
        """Upload any image file"""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select Cephalometric Image", filetypes=filetypes)
        
        if filepath:
            self.image_path = filepath
            image_id = os.path.splitext(os.path.basename(filepath))[0]
            self._set_current_pixel_spacing(image_id)
            self.load_and_display_image(filepath)
            self.analyze_btn.config(state=tk.NORMAL)
            self.gt_landmarks = None
            self.gt_status.config(text="GT: Not available (custom image)")
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.landmarks = None
            self.landmark_listbox.delete(0, tk.END)
            self.mre_label.config(text="MRE: --")
            self.sdr_label.config(text="SDR@2mm: --")
            
    def load_from_test_set(self):
        """Load image from test set with ground truth"""
        test_images_path = os.path.join(self.dataset_path, "test", "Images")
        
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            title="Select Image from Test Set",
            initialdir=test_images_path,
            filetypes=filetypes
        )
        
        if filepath:
            self.image_path = filepath
            self.load_and_display_image(filepath)
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Load ground truth
            self.load_ground_truth(filepath)
            
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.landmarks = None
            self.landmark_listbox.delete(0, tk.END)
            self.mre_label.config(text="MRE: --")
            self.sdr_label.config(text="SDR@2mm: --")

    def load_ground_truth(self, image_path):
        """Load ground truth annotations for the image"""
        # Get image ID from filename
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        self._set_current_pixel_spacing(image_id)
        
        # Find which split this image belongs to
        for split in ['test', 'valid', 'train']:
            senior_path = os.path.join(
                self.dataset_path, split, "Annotations", 
                "Cephalometric Landmarks", "Senior Orthodontists", f"{image_id}.json"
            )
            junior_path = os.path.join(
                self.dataset_path, split, "Annotations",
                "Cephalometric Landmarks", "Junior Orthodontists", f"{image_id}.json"
            )
            
            if os.path.exists(senior_path) and os.path.exists(junior_path):
                try:
                    # Load both annotations
                    with open(senior_path, 'r') as f:
                        senior_data = json.load(f)
                    with open(junior_path, 'r') as f:
                        junior_data = json.load(f)
                    
                    # Extract landmarks in JSON file order (same as dataset.py)
                    senior_coords = [[lm['value']['x'], lm['value']['y']] for lm in senior_data['landmarks']]
                    junior_coords = [[lm['value']['x'], lm['value']['y']] for lm in junior_data['landmarks']]
                    
                    senior_coords = np.array(senior_coords, dtype=np.float32)
                    junior_coords = np.array(junior_coords, dtype=np.float32)
                    
                    # Average the coordinates (exactly like dataset.py does)
                    gt_coords = np.ceil(0.5 * (senior_coords + junior_coords))
                    
                    self.gt_landmarks = gt_coords
                    self.gt_status.config(text=f"GT: Loaded ({split} set)", fg='#00ff00')
                    print(f"Loaded GT from {split} set: {len(gt_coords)} landmarks")
                    return
                    
                except Exception as e:
                    print(f"Error loading GT: {e}")
                    import traceback
                    traceback.print_exc()
                    
        self.gt_landmarks = None
        self.gt_status.config(text="GT: Not found", fg='#ff6666')
        
    def load_and_display_image(self, filepath):
        """Load and display the image"""
        try:
            self.original_image = Image.open(filepath)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            self.display_image_on_canvas(self.original_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def refresh_display(self):
        """Refresh the display with current settings"""
        if self.original_image is not None:
            self.display_image_on_canvas(
                self.original_image, 
                self.landmarks, 
                self.confidences,
                self.gt_landmarks
            )
            
    def display_image_on_canvas(self, image, landmarks=None, confidences=None, gt_landmarks=None):
        """Display image with landmarks"""
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        display_img = image.copy().resize((new_width, new_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(display_img)
        
        # Draw error lines first (so they're behind the points)
        if (self.show_lines.get() and landmarks is not None and gt_landmarks is not None):
            for pred, gt in zip(landmarks, gt_landmarks):
                px, py = int(pred[0] * scale), int(pred[1] * scale)
                gx, gy = int(gt[0] * scale), int(gt[1] * scale)
                draw.line([(px, py), (gx, gy)], fill='#ffff00', width=1)
        
        # Draw ground truth (squares)
        if self.show_ground_truth.get() and gt_landmarks is not None:
            # Track label positions to avoid overlaps
            label_positions = []
            
            for i, gt in enumerate(gt_landmarks):
                x, y = int(gt[0] * scale), int(gt[1] * scale)
                size = 5
                draw.rectangle([x-size, y-size, x+size, y+size], 
                              fill=None, outline='#ff6600', width=2)
                
                # Add landmark number and short name for ground truth if labels enabled
                if self.show_labels.get():
                    # Show only landmark names (no numbers)
                    short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
                    label = short_name
                    
                    # Position label centered above landmark
                    label_width = len(label) * 6  # Estimate text width
                    label_x = x - label_width // 2  # Center horizontally
                    label_y = y - 15  # Move above the point
                    
                    # Check for overlaps and adjust position
                    for pos_x, pos_y, label_width, label_height in label_positions:
                        # Simple overlap detection
                        if (abs(label_x - pos_x) < label_width and 
                            abs(label_y - pos_y) < label_height):
                            # Move label up to avoid overlap
                            label_y = pos_y - label_height - 2
                    
                    # Estimate label size (rough approximation)
                    label_height = 10
                    label_positions.append((label_x, label_y, label_width, label_height))
                    
                    draw.text((label_x, label_y), label, fill='#ff6600')
                else:
                    # Show only numbers when labels are disabled
                    label = str(i+1)
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y - 15
                    
                    # Check for overlaps
                    for pos_x, pos_y, label_width, label_height in label_positions:
                        if (abs(label_x - pos_x) < label_width and 
                            abs(label_y - pos_y) < label_height):
                            label_y = pos_y - label_height - 2
                    
                    label_height = 10
                    label_positions.append((label_x, label_y, label_width, label_height))
                    
                    draw.text((label_x, label_y), label, fill='#ff6600')
        else:
            # Initialize empty label_positions for prediction labels to use
            label_positions = []
        
        # Draw predictions (circles)
        if self.show_predictions.get() and landmarks is not None:
            colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                     '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                     '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                     '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                     '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                     '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
            
            # Track label positions to avoid overlaps
            pred_label_positions = []
            
            for i, lm in enumerate(landmarks):
                x, y = int(lm[0] * scale), int(lm[1] * scale)
                color = colors[i % len(colors)]
                radius = 5
                draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                           fill=color, outline='white', width=1)
                
                # Add landmark number and short name if labels enabled
                if self.show_labels.get():
                    # Show only landmark names (no numbers)
                    short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
                    label = short_name
                    
                    # Position label centered above landmark
                    label_width = len(label) * 6  # Estimate text width
                    label_x = x - label_width // 2  # Center horizontally
                    label_y = y - 15  # Move above the point
                    
                    # Check for overlaps with both GT and prediction labels
                    all_positions = label_positions + pred_label_positions
                    for pos_x, pos_y, label_width, label_height in all_positions:
                        # Simple overlap detection
                        if (abs(label_x - pos_x) < label_width and 
                            abs(label_y - pos_y) < label_height):
                            # Move label up to avoid overlap
                            label_y = pos_y - label_height - 2
                    
                    # Estimate label size (rough approximation)
                    label_height = 10
                    pred_label_positions.append((label_x, label_y, label_width, label_height))
                    
                    draw.text((label_x, label_y), label, fill=color)
                else:
                    # Show only numbers when labels are disabled
                    label = str(i+1)
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y - 15
                    
                    # Check for overlaps with both GT and prediction labels
                    all_positions = label_positions + pred_label_positions
                    for pos_x, pos_y, label_width, label_height in all_positions:
                        if (abs(label_x - pos_x) < label_width and 
                            abs(label_y - pos_y) < label_height):
                            label_y = pos_y - label_height - 2
                    
                    label_height = 10
                    pred_label_positions.append((label_x, label_y, label_width, label_height))
                    
                    draw.text((label_x, label_y), label, fill=color)
        
        self.photo = ImageTk.PhotoImage(display_img)
        self.canvas.delete('all')
        
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        self.display_scale = scale
        
    def preprocess_image(self, image):
        """Preprocess image for model"""
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        orig_h, orig_w = img_np.shape[:2]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)
        
        img_resized = cv2.resize(img_np, (self.config.INPUT_SIZE[1], self.config.INPUT_SIZE[0]))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
        
        return img_tensor, (orig_w, orig_h)
        
    def analyze_image(self):
        """Run model inference"""
        if self.original_image is None:
            return
            
        self.status_label.config(text="Analyzing...")
        self.root.update()
        
        try:
            img_tensor, orig_size = self.preprocess_image(self.original_image)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                pred_heatmaps, _ = self.model(img_tensor)
            
            method = self.coord_method.get()
            if method == 'soft_argmax':
                coords = soft_argmax_2d(pred_heatmaps, temperature=0.05)
            else:
                coords = extract_coordinates_from_heatmaps(pred_heatmaps, method=method)
            
            scale_x = orig_size[0] / self.config.HEATMAP_SIZE[1]
            scale_y = orig_size[1] / self.config.HEATMAP_SIZE[0]
            
            coords = coords.cpu().numpy()[0]
            coords[:, 0] *= scale_x
            coords[:, 1] *= scale_y
            # Confidence threshold removed; we still keep a placeholder confidences array
            confidences = np.ones((coords.shape[0],), dtype=np.float32)

            self.landmarks = coords
            self.confidences = confidences
            
            # Calculate metrics if GT available
            if self.gt_landmarks is not None:
                self.calculate_metrics()
            
            self.display_image_on_canvas(self.original_image, coords, confidences, self.gt_landmarks)
            self.update_landmark_list()

            self.status_label.config(text="Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def calculate_metrics(self):
        """Calculate MRE and SDR comparing predictions to ground truth"""
        if self.landmarks is None or self.gt_landmarks is None:
            return

        pixel_spacing = float(self.current_pixel_spacing)
        
        # Calculate distances
        distances = np.sqrt(np.sum((self.landmarks - self.gt_landmarks) ** 2, axis=1))
        distances_mm = distances * pixel_spacing
        
        # MRE
        mre = np.mean(distances_mm)
        std = np.std(distances_mm)
        
        # SDR@2mm
        sdr_2mm = (distances_mm <= 2.0).mean() * 100
        sdr_25mm = (distances_mm <= 2.5).mean() * 100
        sdr_3mm = (distances_mm <= 3.0).mean() * 100
        sdr_4mm = (distances_mm <= 4.0).mean() * 100
        
        self.mre_label.config(text=f"MRE: {mre:.2f} ± {std:.2f} mm")
        self.sdr_label.config(text=f"SDR: 2mm={sdr_2mm:.1f}% | 2.5={sdr_25mm:.1f}% | 3={sdr_3mm:.1f}% | 4={sdr_4mm:.1f}%")
        
    def update_landmark_list(self):
        """Update landmark listbox with comparison"""
        self.landmark_listbox.delete(0, tk.END)
        
        if self.landmarks is None:
            return
            
        pixel_spacing = float(self.current_pixel_spacing)

        for i, pred in enumerate(self.landmarks):
            short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
            
            if self.gt_landmarks is not None:
                gt = self.gt_landmarks[i]
                dist = np.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2) * pixel_spacing
                status = "✓" if dist <= 2.0 else "○"
                text = f"{status} {i+1:2d}. {short_name:<5} P:({pred[0]:4.0f},{pred[1]:4.0f}) GT:({gt[0]:4.0f},{gt[1]:4.0f}) Err:{dist:.1f}mm"
                
                # Color based on error
                if dist <= 2.0:
                    color = '#00ff00'  # Green - good
                elif dist <= 4.0:
                    color = '#ffff00'  # Yellow - ok
                else:
                    color = '#ff6666'  # Red - bad
            else:
                text = f"  {i+1:2d}. {short_name:<5} ({pred[0]:.0f}, {pred[1]:.0f})"
                color = '#00ff00'
            
            self.landmark_listbox.insert(tk.END, text)
            self.landmark_listbox.itemconfig(i, fg=color)


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TScale', background='#3c3c3c', troughcolor='#1e1e1e')
    
    app = CephalometricGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
