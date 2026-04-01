import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageEnhance, ImageFont
import torch
import numpy as np
import cv2
import os
import sys
import json
import glob
import csv
import math
import io
from datetime import datetime
from pathlib import Path
import threading

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(WEB_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

def get_resource_path(*parts: str) -> str:
    """
    Resolve a resource path both when running from source and when bundled by PyInstaller.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(getattr(sys, "_MEIPASS"))
    else:
        base_path = Path(PROJECT_ROOT)
    return str(base_path.joinpath(*parts))

DEFAULT_MODEL_PATH = get_resource_path("checkpoints", "best_model.pth")
from src.models.hrnet import HRNet
from src.utils.metrics import extract_coordinates_from_heatmaps, soft_argmax_2d
from src.config.config import TrainConfig
from src.utils.dicom_handler import is_dicom_file, load_dicom_as_pil, extract_dicom_pixel_spacing, PYDICOM_AVAILABLE

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
        self.root.title("Cephalometric Landmark Prediction")
        self.root.geometry("1800x950")
        self.root.configure(bg='#e8f4f8')  # Light dental blue background
        
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
        self.landmark_insights = []
        self.predicted_landmarks = None
        self.landmark_overrides = {}
        self.suppressed_landmarks = set()
        self.highlighted_landmark = None
        self.landmark_edit_mode = tk.BooleanVar(value=False)
        self.pending_landmark_set = None
        self._dragging_landmark_index = None
        self.confidence_scores = None
        self.landmark_tree = None
        self._landmark_hover_iid = None
        self._hover_landmark_index = None
        self._landmark_screen_positions = []
        self._suspend_tree_events = False
        # Use default font for tracing labels
        self.tracing_font_large = None
        self.tracing_font_small = None
        self.analysis_types = {
            "Steiner": {
                "SNA": {"normal": "82° ± 2°", "units": "°"},
                "SNB": {"normal": "80° ± 2°", "units": "°"},
                "ANB": {"normal": "2° ± 2°", "units": "°"},
                "Mandibular Plane": {"normal": "32° ± 3°", "units": "°"},
            },
            "Ricketts": {
                "Facial Axis": {"normal": "90° ± 3°", "units": "°"},
                "Facial Depth": {"normal": "90° ± 3°", "units": "°"},
                "Mandibular Plane": {"normal": "26° ± 4°", "units": "°"},
                "Lower Facial Height": {"normal": "47% ± 4%", "units": "%"},
            },
            "McNamara": {
                "Maxillary Length": {"normal": "52mm ± 3mm", "units": "mm"},
                "Mandibular Length": {"normal": "65mm ± 4mm", "units": "mm"},
                "Anterior Facial Height": {"normal": "110mm ± 4mm", "units": "mm"},
            },
        }
        self.analysis_type = tk.StringVar(value="Steiner")
        self.analysis_type.trace_add("write", lambda *_: self._on_analysis_type_change())
        self.show_tracing = tk.BooleanVar(value=True)
        self.measurement_results = {}
        self.analysis_results_tree = None
        self.landmark_index_map = {abbr: idx for idx, abbr in enumerate(LANDMARK_SHORT)}
        # Default tracing overlay segments (used if analysis lacks a custom definition)
        self.tracing_segments = [
            # Basic reference planes
            ("Cranial Base", "S", "N"),
            ("Maxillary Plane", "ANS", "PNS"),
            ("Mandibular Plane", "Go", "Gn"),
            # Facial analysis lines
            ("Facial Axis", "N", "Gn"),
            ("Frankfort Horizontal", "Po", "Or"),
            # Occlusal references
            ("Occlusal Plane", "U6", "L6"),
            ("Bisector Occlusal", "U1", "L1"),
            # Aesthetic lines
            ("E-line", "Prn", "Pog'"),
            ("Nasolabial", "Sn", "Ls"),
            # Additional reference lines
            ("Y-axis", "S", "Gn"),
            ("Palatal Plane", "ANS", "PNS"),
        ]
        # Analysis-specific tracing overlays
        self.analysis_tracing_segments = {
            "Steiner": [
                ("Cranial Base", "S", "N"),
                ("Maxillary Plane", "ANS", "PNS"),
                ("Mandibular Plane", "Go", "Gn"),
                ("Frankfort Horizontal", "Po", "Or"),
                ("Facial Axis", "N", "Gn"),
                ("E-line", "Prn", "Pog'"),
                ("Nasolabial", "Sn", "Ls"),
            ],
            "Ricketts": [
                ("Facial Axis", "N", "Gn"),
                ("Mandibular Plane", "Go", "Gn"),
                ("Lower Facial Height", "ANS", "Me"),
                ("Total Facial Height", "N", "Me"),
                ("Occlusal Plane", "U6", "L6"),
                ("Bisector Occlusal", "U1", "L1"),
                ("Palatal Plane", "ANS", "PNS"),
            ],
            "McNamara": [
                ("Cranial Base", "S", "N"),
                ("Maxillary Length", "Co", "A"),
                ("Mandibular Length", "Co", "Gn"),
                ("Anterior Facial Height", "N", "Me"),
                ("Facial Convexity", "A", "N'"),
                ("Mandibular Plane", "Go", "Gn"),
            ],
        }
        self.analysis_definitions = {
            "Steiner": [
                {"name": "SNA", "type": "angle", "points": ("S", "N", "A"),
                 "units": "°", "normal": "82° ± 2°", "range": (80, 84)},
                {"name": "SNB", "type": "angle", "points": ("S", "N", "B"),
                 "units": "°", "normal": "80° ± 2°", "range": (78, 82)},
                {"name": "ANB", "type": "difference", "a": "SNA", "b": "SNB",
                 "units": "°", "normal": "2° ± 2°", "range": (0, 4)},
                {"name": "Mandibular Plane", "type": "angle_lines",
                 "line1": ("S", "N"), "line2": ("Go", "Gn"),
                 "units": "°", "normal": "32° ± 3°", "range": (29, 35)},
            ],
            "Ricketts": [
                {"name": "Interincisal Angle", "type": "angle_lines",
                 "line1": ("U1A", "U1"), "line2": ("L1A", "L1"),
                 "units": "°", "normal": "130° ± 6°", "range": (124, 136)},
                {"name": "Lower Facial Height %", "type": "ratio",
                 "segments": (("ANS", "Me"), ("N", "Me")),
                 "units": "%", "normal": "55% ± 5%", "range": (50, 60)},
                {"name": "Mandibular Plane Angle", "type": "angle_lines",
                 "line1": ("S", "N"), "line2": ("Go", "Gn"),
                 "units": "°", "normal": "26° ± 4°", "range": (22, 30)},
                {"name": "Facial Depth Angle", "type": "angle_lines",
                 "line1": ("Po", "Or"), "line2": ("N", "Pog"),
                 "units": "°", "normal": "90° ± 3°", "range": (87, 93)},
            ],
            "McNamara": [
                {"name": "Maxillary Length", "type": "distance",
                 "points": ("Co", "A"), "units": "mm",
                 "normal": "52 ± 3 mm", "range": (49, 55)},
                {"name": "Mandibular Length", "type": "distance",
                 "points": ("Co", "Gn"), "units": "mm",
                 "normal": "65 ± 4 mm", "range": (61, 69)},
                {"name": "Anterior Facial Height", "type": "distance",
                 "points": ("N", "Me"), "units": "mm",
                 "normal": "110 ± 4 mm", "range": (106, 114)},
                {"name": "Facial Convexity", "type": "distance",
                 "points": ("A", "N'"), "units": "mm",
                 "normal": "2 ± 2 mm", "range": (0, 4)},
            ],
        }
        self.analysis_results_container = None
        self.analysis_results_tree = None
        self.analysis_results_visible = True
        self.analysis_summary_label = None
        # Image manipulation state
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.invert_colors = tk.BooleanVar(value=False)
        self.zoom_scale = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 4.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._pan_start = None
        self.placeholder_text_id = None
        self._image_draw_params = None
        self.canvas_image_id = None
        self.canvas_bindings_initialized = False
        self.brightness_var = tk.DoubleVar(value=self.brightness_factor)
        self.contrast_var = tk.DoubleVar(value=self.contrast_factor)
        self.brightness_slider = None
        self.contrast_slider = None
        self.image_tools_frame = None
        
        # Dataset path - works for both source and bundled builds
        self.dataset_path = get_resource_path("dataset")
        
        # Theme colors - Dental/Clinical
        self.PANEL_BG = '#ffffff'  # Clean white panels
        self.TEXT_COLOR = '#2c3e50'  # Dark blue-gray text
        self.PREDICTION_COLOR = '#3498db'  # Dental blue
        self.GROUND_TRUTH_COLOR = '#27ae60'  # Medical green
        self.ERROR_LINE_COLOR = '#e74c3c'  # Soft red
        self.ACCENT_COLOR = '#16a085'  # Teal accent

        self._load_pixel_spacing_map()
        
        # Create UI immediately
        self.create_ui()
        
        # Load model in background thread
        self.model = None
        self.model_loaded = False
        self.model_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self.model_thread.start()
        
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
        
    def _load_model_async(self):
        """Load the trained model in background thread"""
        try:
            # Update UI to show loading state
            self.root.after(0, self._on_model_loading_start)
            
            self.model = HRNet(
                num_landmarks=self.config.NUM_LANDMARKS,
                use_multitask=self.config.USE_MULTITASK
            )
            
            checkpoint_path = DEFAULT_MODEL_PATH
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
                self.model_path = None
                self.root.after(0, lambda: messagebox.showwarning(
                    "Model Missing",
                    f"No bundled checkpoint found at:\n{checkpoint_path}\n\n"
                    "Please select a model file manually."
                ))
            
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            print(f"Model loaded on {self.device}")
            
            # Update UI when model is ready
            self.root.after(0, self._on_model_loaded)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {str(e)}"))
            import traceback
            traceback.print_exc()
            self.root.after(0, self._on_model_load_failed)

    def _on_model_loading_start(self):
        """Update UI when model loading starts"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Loading model...")
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.config(state=tk.DISABLED, text="🔄 Loading model...")

    def _on_model_loaded(self):
        """Update UI when model is successfully loaded"""
        self.model_loaded = True
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Ready")
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.config(state=tk.NORMAL, text="🔍 Analyze Image")

    def _on_model_load_failed(self):
        """Update UI when model loading fails"""
        self.model_loaded = False
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Model load failed")
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.config(state=tk.DISABLED, text="🔍 Model unavailable")

    def load_model(self):
        """Legacy method - now handled by async loading"""
        # This method is kept for compatibility but model loading is now async
        pass

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
        """Create a simplified, user-friendly interface"""
        self.root.configure(bg='#e8f4f8')  # Light dental blue background
        self.canvas_bindings_initialized = False
        main_frame = tk.Frame(self.root, bg='#e8f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = tk.Frame(main_frame, bg='#ffffff', width=360, bd=0, relief='flat')
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)
        
        title = tk.Label(
            left_panel,
            text="AI Cephalometric\nAssistant",
            font=('Segoe UI', 16, 'bold'),
            bg='#ffffff',
            fg='#1f2933',
            justify=tk.LEFT
        )
        title.pack(anchor=tk.W, padx=20, pady=(15, 5))
        
        subtitle = tk.Label(
            left_panel,
            text="Upload a lateral cephalogram and let the model detect all 29 landmarks automatically.",
            font=('Segoe UI', 12),
            bg='#ffffff',
            fg='#52606d',
            wraplength=300,
            justify=tk.LEFT
        )
        subtitle.pack(anchor=tk.W, padx=20, pady=(0, 15))
        
        btn_frame = tk.Frame(left_panel, bg='#ffffff')
        btn_frame.pack(fill=tk.X, padx=20, pady=(10, 5))
        
        self.upload_btn = tk.Button(
            btn_frame,
            text="📂 Select X-ray",
            font=('Segoe UI', 10, 'bold'),
            bg='#3498db',  # Dental blue
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.upload_image
        )
        self.upload_btn.pack(fill=tk.X, pady=2)
        
        self.analyze_btn = tk.Button(
            btn_frame,
            text="🔍 Analyze Image",
            font=('Segoe UI', 10, 'bold'),
            bg='#16a085',  # Teal accent
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(fill=tk.X, pady=4)
        
        self.export_btn = tk.Button(
            btn_frame,
            text="📄 Export Report",
            font=('Segoe UI', 10, 'bold'),
            bg='#27ae60',  # Medical green
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.show_export_dialog,
            state=tk.DISABLED
        )
        self.export_btn.pack(fill=tk.X, pady=2)
        
        # Status and feedback labels
        self.status_label = tk.Label(left_panel, text="Ready", font=('Segoe UI', 9),
                                     bg='#ffffff', fg='#6b7280')
        self.status_label.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Hidden labels retained for callbacks
        self.gt_status = tk.Label(left_panel, text="GT: Not available (custom image)",
                                  font=('Segoe UI', 9), bg='#ffffff', fg='#6b7280', justify=tk.LEFT)
        self.gt_hint_label = tk.Label(
            left_panel,
            text="Ground truth: not loaded",
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg='#6b7280',
            justify=tk.LEFT,
            wraplength=260
        )
        
        # Analysis results label (packed when needed)
        self.analysis_result_label = tk.Label(
            left_panel,
            text="No analysis yet.",
            font=('Segoe UI', 11, 'bold'),
            bg='#ffffff',
            fg='#111827',
            justify=tk.LEFT,
            wraplength=260
        )
        
        # Summary label (packed when needed)
        self.summary_label = tk.Label(
            left_panel,
            text="Run an analysis to view landmark accuracy.",
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg='#52606d',
            justify=tk.LEFT,
            wraplength=260
        )
        
        # Landmark Legend Section
        legend_frame = tk.LabelFrame(left_panel, text="Landmark Legend", 
                                     bg='#ffffff', fg='#1f2937', font=('Segoe UI', 10, 'bold'))
        legend_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Create scrollable legend content
        legend_canvas = tk.Canvas(legend_frame, bg='#ffffff', highlightthickness=0)
        scrollbar = tk.Scrollbar(legend_frame, orient="vertical", command=legend_canvas.yview)
        scrollable_frame = tk.Frame(legend_canvas, bg='#ffffff')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox("all"))
        )
        
        legend_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        legend_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add landmark legend items with prediction colors
        landmark_names = [
            ("A", "A Point"), ("ANS", "Anterior Nasal Spine"), ("B", "B Point"), ("Me", "Menton"),
            ("N", "Nasion"), ("Or", "Orbitale"), ("Pog", "Pogonion"), ("PNS", "Posterior Nasal Spine"),
            ("Prn", "Pronasale"), ("R", "Ricketts' Point"), ("S", "Sella"), ("Ar", "Articulare"),
            ("Co", "Condylion"), ("Gn", "Gnathion"), ("Go", "Gonion"), ("Po", "Porion"),
            ("L5", "Lower First Molar"), ("L1", "Lower Central Incisor"), ("L6", "Lower First Premolar"),
            ("U5", "Upper First Molar"), ("U1A", "Upper Central Incisor Apex"), ("U1", "Upper Central Incisor Edge"),
            ("U6", "Upper First Premolar"), ("L1A", "Lower Central Incisor Apex"), ("Li", "Lower Incisor Edge"),
            ("Ls", "Lower Lip"), ("N'", "Soft Tissue Nasion"), ("Pog'", "Soft Tissue Pogonion"), ("Sn", "Subnasale")
        ]
        
        # Prediction colors array (same as display)
        colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                  '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                  '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                  '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                  '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                  '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
        
        for i, (abbr, full_name) in enumerate(landmark_names):
            row_frame = tk.Frame(scrollable_frame, bg='#ffffff')
            row_frame.pack(fill=tk.X, padx=5, pady=1)
            
            # Landmark abbreviation with prediction color
            color = colors[i % len(colors)]
            abbr_label = tk.Label(row_frame, text=abbr, font=('Segoe UI', 9, 'bold'),
                                bg='#ffffff', fg=color, width=4, anchor='w')
            abbr_label.pack(side=tk.LEFT, padx=(5, 2))
            
            # Full name
            name_label = tk.Label(row_frame, text=full_name, font=('Segoe UI', 9),
                                bg='#ffffff', fg='#52606d', anchor='w')
            name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        legend_canvas.pack(side="left", fill="both", expand=True, padx=(5, 0))
        scrollbar.pack(side="right", fill="y")
        
        preview_panel = tk.Frame(main_frame, bg='#ffffff', bd=0, relief='flat')
        preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Single view canvas
        canvas_wrapper = tk.Frame(preview_panel, bg='#ffffff')
        canvas_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_wrapper, bg='#111827', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._setup_canvas_bindings()
        self._show_placeholder()
        self._build_image_tools(canvas_wrapper)
        self._build_landmark_panel(preview_panel)
        self._update_export_state()
        
    def analyze_image(self):
        """Analyze the current image with the model"""
        if not self.model_loaded or self.model is None:
            messagebox.showinfo("Model Loading", "Please wait for the model to finish loading...")
            return
        
        if self.original_image is None:
            messagebox.showinfo("No Image", "Please upload an X-ray image first.")
            return
        
        # Show loading status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Analyzing...")
        
        # Run prediction in background thread to keep UI responsive
        def run_analysis():
            try:
                # Convert image to tensor and run prediction
                image_tensor, orig_size = self._preprocess_image(self.original_image)
                
                with torch.no_grad():
                    pred_heatmaps, cvm_logits = self.model(image_tensor)
                
                # Extract coordinates from heatmaps
                if self.coord_method.get() == 'soft_argmax':
                    coords = soft_argmax_2d(pred_heatmaps, temperature=0.05)
                else:
                    coords = extract_coordinates_from_heatmaps(pred_heatmaps)
                
                # Scale heatmap coordinates back to original image size
                scale_x = orig_size[0] / self.config.HEATMAP_SIZE[1]
                scale_y = orig_size[1] / self.config.HEATMAP_SIZE[0]
                
                coords = coords.cpu().numpy()[0]
                coords[:, 0] *= scale_x
                coords[:, 1] *= scale_y
                
                landmarks = torch.from_numpy(coords)
                
                # Update UI with results
                self.root.after(0, lambda: self._on_analysis_complete(landmarks))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._on_analysis_error(str(e)))
        
        # Start analysis thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def _on_analysis_complete(self, landmarks):
        """Handle successful analysis completion"""
        try:
            self.predicted_landmarks = landmarks.cpu().numpy().squeeze()
            self.landmarks = self._compose_landmarks()
            self.confidence_scores = None  # Could be added later
            
            # Update UI
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Analysis complete")
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.config(state=tk.NORMAL)
            if hasattr(self, 'export_btn'):
                self.export_btn.config(state=tk.NORMAL)
            
            # Refresh display and compute measurements
            self.refresh_display()
            self.measurement_results = self._compute_measurements()
            self._update_analysis_results_table()
            self._update_export_state()
            
        except Exception as e:
            self._on_analysis_error(str(e))
    
    def _on_analysis_error(self, error_msg):
        """Handle analysis error"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Analysis failed")
        messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{error_msg}")
    
    def _preprocess_image(self, image):
        """Preprocess image for model input (must match training pipeline exactly)
        
        Training pipeline: grayscale -> CLAHE -> resize 864x768 -> normalize [0,1] -> (1,1,H,W)
        
        Returns:
            img_tensor: (1, 1, H, W) float32 tensor on self.device
            orig_size: (orig_w, orig_h) tuple for scaling coordinates back
        """
        img_np = np.array(image)
        
        # Convert to grayscale (model expects 1 channel)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        orig_h, orig_w = img_np.shape[:2]
        
        # Apply CLAHE (same as training: clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)
        
        # Resize to model input size (W, H) = (768, 864)
        img_resized = cv2.resize(img_np, (self.config.INPUT_SIZE[1], self.config.INPUT_SIZE[0]))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Create (1, 1, H, W) tensor
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
        
        if hasattr(self, 'device'):
            img_tensor = img_tensor.to(self.device)
        
        return img_tensor, (orig_w, orig_h)

    def show_export_dialog(self):
        """Show export options dialog"""
        if self.landmarks is None:
            messagebox.showinfo("No Data", "Please analyze an image first before exporting.")
            return
        
        # Create export dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Options")
        dialog.geometry("500x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")
        
        # Main frame
        main_frame = tk.Frame(dialog, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(
            main_frame,
            text="Export Analysis Results",
            font=('Segoe UI', 14, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        ).pack(pady=(0, 20))
        
        # Export options
        options_frame = tk.Frame(main_frame, bg='#ffffff')
        options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create checkbox variables
        self.export_overlay_var = tk.BooleanVar(value=True)
        self.export_clean_var = tk.BooleanVar(value=True)
        self.export_pdf_var = tk.BooleanVar(value=REPORTLAB_AVAILABLE)
        
        # PNG Export Options
        png_frame = tk.Frame(options_frame, bg='#f8fafc', relief=tk.RIDGE, bd=1)
        png_frame.pack(fill=tk.X, pady=(0, 10))
        
        overlay_check = tk.Checkbutton(
            png_frame,
            text="🖼️ PNG with Landmarks + Tracing Overlay",
            font=('Segoe UI', 11),
            bg='#f8fafc',
            fg='#1f2937',
            selectcolor='#ffffff',
            activebackground='#f8fafc',
            variable=self.export_overlay_var
        )
        overlay_check.pack(side=tk.LEFT, padx=15, pady=10)
        
        png_clean_frame = tk.Frame(options_frame, bg='#f8fafc', relief=tk.RIDGE, bd=1)
        png_clean_frame.pack(fill=tk.X, pady=(0, 10))
        
        clean_check = tk.Checkbutton(
            png_clean_frame,
            text="🖼️ PNG with Landmarks Only (No Tracing Lines)",
            font=('Segoe UI', 11),
            bg='#f8fafc',
            fg='#1f2937',
            selectcolor='#ffffff',
            activebackground='#f8fafc',
            variable=self.export_clean_var
        )
        clean_check.pack(side=tk.LEFT, padx=15, pady=10)
        
        # PDF Export Option
        pdf_frame = tk.Frame(options_frame, bg='#f8fafc', relief=tk.RIDGE, bd=1)
        pdf_frame.pack(fill=tk.X, pady=(0, 10))
        
        pdf_state = tk.NORMAL if REPORTLAB_AVAILABLE else tk.DISABLED
        pdf_check = tk.Checkbutton(
            pdf_frame,
            text="📄 PDF Clinical Report",
            font=('Segoe UI', 11),
            bg='#f8fafc',
            fg='#1f2937',
            selectcolor='#ffffff',
            activebackground='#f8fafc',
            variable=self.export_pdf_var,
            state=pdf_state
        )
        pdf_check.pack(side=tk.LEFT, padx=15, pady=10)
        if not REPORTLAB_AVAILABLE:
            tk.Label(
                pdf_frame,
                text="(Install 'reportlab' to enable PDF export: pip install reportlab)",
                font=('Segoe UI', 9),
                bg='#f8fafc',
                fg='#a94442'
            ).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#ffffff')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        tk.Button(
            button_frame,
            text="Export Selected",
            font=('Segoe UI', 11, 'bold'),
            bg='#27ae60',  # Medical green
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=lambda: self.export_selected(dialog)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            button_frame,
            text="Cancel",
            font=('Segoe UI', 11),
            bg='#e74c3c',  # Soft red
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=dialog.destroy
        ).pack(side=tk.RIGHT)
    
    def export_selected(self, dialog):
        """Export only the selected options"""
        try:
            # Check if any options are selected
            pdf_selected = self.export_pdf_var.get()
            if not (self.export_overlay_var.get() or self.export_clean_var.get() or pdf_selected):
                messagebox.showwarning("No Selection", "Please select at least one export option.")
                return
            
            export_dir = self._get_export_directory()
            if not export_dir:
                return
            
            base_filename = self._generate_filename()
            exported_files = []
            
            # Export selected options
            if self.export_overlay_var.get():
                self.export_png_with_overlay(export_dir, base_filename)
                exported_files.append(f"{base_filename}_with_overlay.png")
            
            if self.export_clean_var.get():
                self.export_png_clean(export_dir, base_filename)
                exported_files.append(f"{base_filename}_clean.png")
            
            if pdf_selected:
                if REPORTLAB_AVAILABLE:
                    self.export_pdf_report(export_dir, base_filename)
                    exported_files.append(f"{base_filename}_report.pdf")
                else:
                    # Fallback to text report if ReportLab isn't installed
                    self._export_simple_text(export_dir, base_filename)
                    exported_files.append(f"{base_filename}_report.txt")
            
            # Show success message
            if len(exported_files) == 1:
                messagebox.showinfo("Export Complete", f"File exported to:\n{export_dir}")
            else:
                messagebox.showinfo("Export Complete", f"Files exported to:\n{export_dir}\n\nExported:\n" + "\n".join(exported_files))
            
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export files:\n{str(e)}")
    
    def _get_export_directory(self):
        """Get export directory from user"""
        return filedialog.askdirectory(title="Select Export Directory")
    
    def _generate_filename(self):
        """Generate base filename with timestamp"""
        base_name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "analysis"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    def export_png_with_overlay(self, export_dir, base_filename):
        """Export PNG image with original image size and overlays"""
        try:
            # Get current display parameters
            if not hasattr(self, '_image_draw_params') or self.original_image is None:
                print("No image data available for export")
                return
            
            # Get original image dimensions
            orig_w = self.original_image.width
            orig_h = self.original_image.height
            
            # Create image matching original image size
            export_img = Image.new('RGB', (orig_w, orig_h), '#111827')
            draw = ImageDraw.Draw(export_img)
            
            # Apply image adjustments
            adjusted_img = self._apply_image_adjustments(self.original_image)
            
            # Paste the adjusted image (full size)
            export_img.paste(adjusted_img, (0, 0))
            
            # Draw landmarks on the image (using original image coordinates)
            if self.landmarks is not None and self.show_predictions.get():
                # Use the same color scheme as the actual display
                colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                         '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                         '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                         '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                         '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                         '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
                
                for i, landmark in enumerate(self.landmarks):
                    if landmark is not None and np.all(np.isfinite(landmark)):
                        # Use original landmark coordinates (no scaling needed)
                        landmark_x = int(landmark[0])
                        landmark_y = int(landmark[1])
                        
                        # Use the same color as display
                        color = colors[i % len(colors)]
                        
                        # Highlight if selected
                        if i == self.highlighted_landmark:
                            color = '#FF0000'
                        
                        # Draw landmark (same radius as display: 5)
                        radius = 5
                        draw.ellipse([landmark_x-radius, landmark_y-radius, landmark_x+radius, landmark_y+radius], 
                                   fill=color, outline='#000000', width=1)
            
            # Draw tracing if enabled (using original coordinates)
            if self.show_tracing.get():
                self._draw_tracing_overlays(draw, 1.0, 0, 0)
            
            # Draw tracing labels directly for export (regenerate them for original image coordinates)
            if self.show_tracing.get():
                self._draw_tracing_labels_for_export(draw, 1.0, 0, 0, orig_w, orig_h, font_size=26)
            
            # Draw landmark labels (like the ones shown on hover)
            if self.landmarks is not None and self.show_predictions.get():
                self._draw_landmark_labels_for_export(draw, 1.0, orig_w, orig_h)
            
            # Save the export image
            img_path = os.path.join(export_dir, f"{base_filename}_with_overlay.png")
            export_img.save(img_path, "PNG", dpi=(300, 300))
            
        except Exception as e:
            print(f"PNG overlay export error: {e}")
            import traceback
            traceback.print_exc()
    
    def export_png_clean(self, export_dir, base_filename):
        """Export PNG image with landmarks but without tracing overlay (original image size)"""
        try:
            # Get current display parameters
            if not hasattr(self, '_image_draw_params') or self.original_image is None:
                print("No image data available for export")
                return
            
            # Get original image dimensions
            orig_w = self.original_image.width
            orig_h = self.original_image.height
            
            # Create image matching original image size
            export_img = Image.new('RGB', (orig_w, orig_h), '#111827')
            draw = ImageDraw.Draw(export_img)
            
            # Apply image adjustments
            adjusted_img = self._apply_image_adjustments(self.original_image)
            
            # Paste the adjusted image (full size)
            export_img.paste(adjusted_img, (0, 0))
            
            # Draw landmarks on the image (using original image coordinates) - NO TRACING
            if self.landmarks is not None and self.show_predictions.get():
                # Use the same color scheme as the actual display
                colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                         '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                         '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                         '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                         '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                         '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
                
                for i, landmark in enumerate(self.landmarks):
                    if landmark is not None and np.all(np.isfinite(landmark)):
                        # Use original landmark coordinates (no scaling needed)
                        landmark_x = int(landmark[0])
                        landmark_y = int(landmark[1])
                        
                        # Use the same color as display
                        color = colors[i % len(colors)]
                        
                        # Highlight if selected
                        if i == self.highlighted_landmark:
                            color = '#FF0000'
                        
                        # Draw landmark (same radius as display: 5)
                        radius = 5
                        draw.ellipse([landmark_x-radius, landmark_y-radius, landmark_x+radius, landmark_y+radius], 
                                   fill=color, outline='#000000', width=1)
            
            # Draw landmark labels (like the ones shown on hover) - NO TRACING
            if self.landmarks is not None and self.show_predictions.get():
                self._draw_landmark_labels_for_export(draw, 1.0, orig_w, orig_h)
            
            # Save the export image
            img_path = os.path.join(export_dir, f"{base_filename}_clean.png")
            export_img.save(img_path, "PNG", dpi=(300, 300))
            
        except Exception as e:
            print(f"PNG clean export error: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_tracing_labels_for_export(self, draw, scale, offset_x, offset_y, img_w, img_h, font_size=24):
        """Draw tracing labels directly for export using original image coordinates"""
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        if font is None:
            return
        
        # Define colors for different types of lines
        line_colors = {
            "Cranial Base": "#FF6B6B",
            "Maxillary Plane": "#4ECDC4", 
            "Palatal Plane": "#4ECDC4",
            "Mandibular Plane": "#45B7D1",
            "Facial Axis": "#96CEB4",
            "Frankfort Horizontal": "#FFEAA7",
            "Occlusal Plane": "#DDA0DD",
            "Bisector Occlusal": "#DDA0DD",
            "E-line": "#FFB6C1",
            "Nasolabial": "#FFB6C1",
            "Y-axis": "#98D8C8",
        }
        
        default_color = "#00e0ff"
        
        current_segments = self.analysis_tracing_segments.get(
            self.analysis_type.get(),
            self.tracing_segments,
        )

        for name, start_abbr, end_abbr in current_segments:
            start = self._get_landmark_point(start_abbr)
            end = self._get_landmark_point(end_abbr)
            if start is None or end is None:
                continue
            
            # Use original image coordinates (already scaled)
            x1 = int(start[0] * scale)
            y1 = int(start[1] * scale)
            x2 = int(end[0] * scale)
            y2 = int(end[1] * scale)
            
            # Calculate label position in original image coordinates
            screen_mid_x = (x1 + x2) // 2
            screen_mid_y = (y1 + y2) // 2
            
            # Adjust label position for Palatal Plane to avoid overlap with Maxillary Plane
            if name == "Palatal Plane":
                # Position Palatal Plane slightly above center
                label_x = screen_mid_x + 30
                label_y = screen_mid_y - 15
            elif name == "Maxillary Plane":
                # Position Maxillary Plane slightly below center
                label_x = screen_mid_x + 30
                label_y = screen_mid_y + 15
            else:
                # Default positioning for other labels
                label_x = screen_mid_x + 4
                label_y = screen_mid_y
            
            # Ensure coordinates are within image bounds
            label_x = max(0, min(label_x, img_w - 50))
            label_y = max(0, min(label_y, img_h - 20))
            
            # Get color
            color = line_colors.get(name, default_color)
            
            # Draw label
            draw.text((label_x, label_y), name, fill=color, font=font)
    
    def _draw_landmark_labels_for_export(self, draw, scale, img_w, img_h):
        """Draw landmark labels for export using the same logic as the on-screen view."""
        if self.landmarks is None:
            return
        
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        if font is None:
            return
        
        show_full_labels = self.show_labels.get()
        colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                  '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                  '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                  '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                  '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                  '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
        
        label_positions = []
        for i, landmark in enumerate(self.landmarks):
            if landmark is None or not np.all(np.isfinite(landmark)):
                continue
            
            label = LANDMARK_SHORT[i] if show_full_labels and i < len(LANDMARK_SHORT) else str(i + 1)
            color = colors[i % len(colors)]
            
            landmark_x = int(landmark[0] * scale)
            landmark_y = int(landmark[1] * scale)
            
            # Measure label to determine width/height
            bbox = draw.textbbox((0, 0), label, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            
            # Match on-screen placement (above by default, below for specific landmarks)
            label_x = landmark_x - label_width // 2
            if show_full_labels and i in (16, 18):
                label_y = landmark_y + 15  # Increased offset to avoid circle overlap
            else:
                label_y = landmark_y - (label_height + 8)  # Increased offset to avoid circle overlap
            
            # Avoid overlapping with previously placed labels (simple vertical adjustments)
            for pos_x, pos_y, width, height in label_positions:
                horizontal_overlap = abs(label_x - pos_x) < (width // 2 + label_width // 2)
                vertical_overlap = abs(label_y - pos_y) < (height + label_height) // 2
                if horizontal_overlap and vertical_overlap:
                    label_y = pos_y - height - 4
            
            label_x = max(0, min(label_x, img_w - label_width))
            label_y = max(0, min(label_y, img_h - label_height))
            
            label_positions.append((label_x, label_y, label_width, label_height))
            draw.text((label_x, label_y), label, fill=color, font=font)
    
    def _create_export_image(self, *, force_tracing=False, tracing_label_font=24):
        """Create high-resolution image with tracing overlay for export"""
        if self.original_image is None:
            return None
        
        # Create high-resolution copy
        export_img = self.original_image.copy()
        
        # Apply adjustments
        export_img = self._apply_image_adjustments(export_img)
        
        # Draw landmarks
        draw = ImageDraw.Draw(export_img)
        
        if self.landmarks is not None:
            for i, landmark in enumerate(self.landmarks):
                if landmark is not None and np.all(np.isfinite(landmark)):
                    x, y = int(landmark[0]), int(landmark[1])
                    
                    # Determine color
                    if i == self.highlighted_landmark:
                        color = '#FF0000'
                    elif i in self.landmark_overrides:
                        color = '#FFA500'
                    elif i in self.suppressed_landmarks:
                        continue  # Skip suppressed landmarks
                    else:
                        color = '#00FF00'
                    
                    # Draw landmark
                    draw.ellipse([x-6, y-6, x+6, y+6], fill=color, outline='#000000', width=2)
                    
                    # Add label
                    if i < len(LANDMARK_SHORT):
                        label = LANDMARK_SHORT[i]
                        draw.text((x+10, y-10), label, fill='#000000', font=self._get_font(12))
        
        # Draw tracing if enabled or forced
        show_tracing = self.show_tracing.get() or force_tracing
        if show_tracing:
            self._draw_tracing_overlays(draw, 1.0, 0, 0)
            self._draw_tracing_labels_for_export(
                draw,
                1.0,
                0,
                0,
                export_img.width,
                export_img.height,
                font_size=tracing_label_font,
            )
        
        return export_img
    
    def export_pdf_report(self, export_dir, base_filename):
        """Generate a PDF report that includes the annotated image and measurement table."""
        if not REPORTLAB_AVAILABLE:
            self._export_simple_text(export_dir, base_filename)
            return
        
        if self.measurement_results is None or not self.measurement_results:
            self.measurement_results = self._compute_measurements()
        
        pdf_path = os.path.join(export_dir, f"{base_filename}_report.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        page_width, page_height = letter
        margin = 48
        y = page_height - margin
        
        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, "Cephalometric Analysis Report")
        y -= 24
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        y -= 16
        c.drawString(margin, y, f"Analysis Type: {self.analysis_type.get()}")
        y -= 16
        img_name = os.path.basename(self.image_path) if self.image_path else "N/A"
        c.drawString(margin, y, f"Image File: {img_name}")
        y -= 16
        c.drawString(margin, y, f"Pixel Spacing: {self.current_pixel_spacing:.3f} mm/pixel")
        y -= 24
        
        # Annotated image
        export_img = self._create_export_image(force_tracing=True, tracing_label_font=30)
        if export_img is not None:
            max_img_width = page_width - (2 * margin)
            max_img_height = page_height * 0.45
            scale = min(max_img_width / export_img.width, max_img_height / export_img.height, 1.0)
            img_width = export_img.width * scale
            img_height = export_img.height * scale
            buffer = io.BytesIO()
            export_img.save(buffer, format="PNG")
            buffer.seek(0)
            image_reader = ImageReader(buffer)
            c.drawImage(
                image_reader,
                margin,
                y - img_height,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True,
                mask='auto'
            )
            # Add extra breathing room between image and table
            y = y - img_height - 72
        
        # Measurements section as a table
        results = self.measurement_results or {}
        rows = list(results.items())
        
        # Ensure enough space for the full table; otherwise start a new page
        estimated_table_height = 24 + 20 + (len(rows) * 18) + 30  # title + header + rows + footer gap
        if y <= margin + estimated_table_height:
            c.showPage()
            y = page_height - margin
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Clinical Measurements")
        y -= 24
        c.setFont("Helvetica", 10)
        
        def ensure_space(current_y, required=60):
            if current_y <= margin + required:
                c.showPage()
                new_y = page_height - margin
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, new_y, "Clinical Measurements (cont.)")
                new_y -= 18
                c.setFont("Helvetica", 10)
                return new_y
            return current_y
        
        if not results:
            c.drawString(margin, y, "No measurements available.")
            y -= 14
        else:
            # Table headers
            table_y = y
            header_height = 20
            row_height = 18
            col_widths = [250, 120, 120]
            col_x = [margin, margin + col_widths[0], margin + col_widths[0] + col_widths[1]]
            
            # Draw header background
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(col_x[0], table_y - header_height, sum(col_widths), header_height, fill=1, stroke=1)
            
            # Draw header text (vertically centered)
            header_base_y = (table_y - header_height) + (header_height / 2) - 3
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(col_x[0] + 4, header_base_y, "Measurement")
            c.drawString(col_x[1] + 4, header_base_y, "Value")
            c.drawString(col_x[2] + 4, header_base_y, "Reference Range")
            
            # Draw rows
            y = table_y - header_height
            rows = list(results.items())
            c.setFont("Helvetica", 10)
            row_line_width = 0.5
            for idx, (metric, data) in enumerate(rows):
                y = ensure_space(y, row_height + 10)
                y -= row_height
                value = data.get("value")
                value_str = self._format_measurement_value(value, data["units"]) if value is not None else "--"
                normal = data.get("normal", "--")
                
                # Draw row background (alternating)
                if idx % 2 == 0:
                    c.setFillColorRGB(0.95, 0.95, 0.95)
                    c.rect(col_x[0], y, sum(col_widths), row_height, fill=1, stroke=0)
                
                # Draw row border
                c.setStrokeColorRGB(0.7, 0.7, 0.7)
                c.setLineWidth(row_line_width)
                c.line(col_x[0], y, col_x[0] + sum(col_widths), y)
                
                # Draw row text - properly aligned within row height
                c.setFillColorRGB(0, 0, 0)
                text_y = y + (row_height / 2) - 3  # Center text vertically in row
                c.drawString(col_x[0] + 4, text_y, metric)
                c.drawString(col_x[1] + 4, text_y, value_str)
                c.drawString(col_x[2] + 4, text_y, normal)
            
            # Draw table border
            c.setStrokeColorRGB(0, 0, 0)
            border_line_width = 0.75
            c.setLineWidth(border_line_width)
            total_height = header_height + (len(rows) * row_height)
            top_y = table_y
            bottom_y = table_y - total_height
            c.rect(col_x[0], table_y - header_height, sum(col_widths), total_height, fill=0, stroke=1)
            
            # Column dividers for alignment clarity
            for divider_x in col_x[1:]:
                c.line(divider_x, top_y, divider_x, bottom_y)
            
            # Restore default line width
            c.setLineWidth(1)
            
            # Add breathing room below the table before footer
            y -= 18
        
        # Footer
        y = ensure_space(y, 30)
        c.setFont("Helvetica", 9)
        c.drawString(margin, y, "Report generated by Cephalometric AI Assistant")
        c.save()
    
    def _get_font(self, size):
        """Get font for drawing text"""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.load_default()
            except:
                return None
    
    def _export_simple_text(self, export_dir, base_filename):
        """Fallback text export if PDF library not available"""
        txt_path = os.path.join(export_dir, f"{base_filename}_report.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("CEPHALOMETRIC ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Analysis Type: {self.analysis_type.get()}\n")
            f.write(f"Image File: {os.path.basename(self.image_path) if self.image_path else 'N/A'}\n")
            f.write(f"Pixel Spacing: {self.current_pixel_spacing:.3f} mm/pixel\n\n")
            
            f.write("CLINICAL MEASUREMENTS\n")
            f.write("-" * 30 + "\n")
            
            if self.measurement_results:
                for metric, data in self.measurement_results.items():
                    value = data.get("value", "--")
                    if value is not None:
                        value = self._format_measurement_value(value, data["units"])
                    f.write(f"{metric}: {value} (Reference: {data['normal']})\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Report generated by Cephalometric Analysis System\n")
    
    def update_conf_label(self, value=None):
        # Kept for backwards compatibility if referenced elsewhere.
        return
    
    def _on_canvas_resize(self, event=None):
        """Keep placeholder text centered when no image is loaded."""
        if not getattr(self, 'placeholder_text_id', None):
            return
        if self.image_path is not None:
            return
        width = event.width if event else self.canvas.winfo_width()
        height = event.height if event else self.canvas.winfo_height()
        self.canvas.coords(self.placeholder_text_id, width / 2, height / 2)
    
    def _build_image_tools(self, parent):
        """Create floating controls for brightness, contrast, invert, reset."""
        self.image_tools_frame = tk.Frame(parent, bg='#1f2933', bd=1, relief='ridge')
        self.image_tools_frame.place(relx=1.0, rely=0.02, anchor='ne')
        
        tk.Label(
            self.image_tools_frame,
            text="Image Tools",
            font=('Segoe UI', 9, 'bold'),
            bg='#1f2933',
            fg='#f3f4f6'
        ).pack(fill=tk.X, padx=8, pady=(6, 4))
        
        tk.Label(
            self.image_tools_frame,
            text="Brightness",
            font=('Segoe UI', 8),
            bg='#1f2933',
            fg='#d1d5db'
        ).pack(anchor=tk.W, padx=8)
        self.brightness_slider = tk.Scale(
            self.image_tools_frame,
            from_=0.3,
            to=1.7,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=150,
            variable=self.brightness_var,
            command=self._on_window_change,
            bg='#1f2933',
            fg='#f9fafb',
            highlightthickness=0,
            troughcolor='#374151'
        )
        self.brightness_slider.pack(padx=8, pady=(0, 6))
        
        tk.Label(
            self.image_tools_frame,
            text="Contrast",
            font=('Segoe UI', 8),
            bg='#1f2933',
            fg='#d1d5db'
        ).pack(anchor=tk.W, padx=8)
        self.contrast_slider = tk.Scale(
            self.image_tools_frame,
            from_=0.3,
            to=1.7,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=150,
            variable=self.contrast_var,
            command=self._on_window_change,
            bg='#1f2933',
            fg='#f9fafb',
            highlightthickness=0,
            troughcolor='#374151'
        )
        self.contrast_slider.pack(padx=8, pady=(0, 6))
        
        tk.Checkbutton(
            self.image_tools_frame,
            text="Invert colors",
            variable=self.invert_colors,
            command=self._on_window_change,
            bg='#1f2933',
            fg='#f9fafb',
            selectcolor='#111827',
            activebackground='#1f2933'
        ).pack(anchor=tk.W, padx=8, pady=(4, 6))
        
        tk.Button(
            self.image_tools_frame,
            text="Reset View",
            command=self._reset_image_view,
            font=('Segoe UI', 9, 'bold'),
            bg='#2563eb',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2'
        ).pack(fill=tk.X, padx=8, pady=(0, 8))
    
    def _build_landmark_panel(self, parent):
        """Create the landmark management sidebar."""
        panel = tk.Frame(parent, bg='#f3f4f6', width=320, bd=0, relief='flat')
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)
        
        header = tk.Label(
            panel,
            text="Landmark Review",
            font=('Segoe UI', 13, 'bold'),
            bg='#f3f4f6',
            fg='#111827'
        )
        header.pack(anchor=tk.W, padx=16, pady=(16, 4))
        
        edit_toggle = tk.Checkbutton(
            panel,
            text="Enable landmark editing",
            variable=self.landmark_edit_mode,
            command=self._toggle_landmark_edit_mode,
            bg='#f3f4f6',
            fg='#111827',
            activebackground='#f3f4f6',
            selectcolor='#e2e8f0'
        )
        edit_toggle.pack(anchor=tk.W, padx=16)
        
        self.landmark_status_label = tk.Label(
            panel,
            text="Manual overrides: none",
            font=('Segoe UI', 9),
            bg='#f3f4f6',
            fg='#4b5563',
            wraplength=280,
            justify=tk.LEFT
        )
        self.landmark_status_label.pack(fill=tk.X, padx=16, pady=(4, 6))
        
        tree_container = tk.Frame(panel, bg='#f3f4f6')
        tree_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        
        columns = ('abbr', 'status')
        self.landmark_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show='headings',
            height=18,
            selectmode='browse'
        )
        self.landmark_tree.heading('abbr', text='Landmark')
        self.landmark_tree.heading('status', text='Status')
        self.landmark_tree.column('abbr', width=190, anchor=tk.CENTER)
        self.landmark_tree.column('status', width=80, anchor=tk.CENTER)
        
        tree_scroll = ttk.Scrollbar(tree_container, orient='vertical', command=self.landmark_tree.yview)
        self.landmark_tree.configure(yscrollcommand=tree_scroll.set)
        self.landmark_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.landmark_tree.bind('<<TreeviewSelect>>', self._on_landmark_select)

        analysis_controls = tk.Frame(panel, bg='#f3f4f6')
        analysis_controls.pack(fill=tk.X, padx=12, pady=(4, 4))
        tk.Label(
            analysis_controls,
            text="Analysis Type",
            font=('Segoe UI', 9, 'bold'),
            bg='#f3f4f6',
            fg='#1f2933'
        ).pack(anchor=tk.W)
        self.analysis_combo = ttk.Combobox(
            analysis_controls,
            textvariable=self.analysis_type,
            values=list(self.analysis_definitions.keys()),
            state='readonly'
        )
        self.analysis_combo.pack(fill=tk.X, pady=(2, 6))
        tk.Checkbutton(
            analysis_controls,
            text="Show tracing overlay",
            variable=self.show_tracing,
            command=self._on_tracing_toggle,
            bg='#f3f4f6',
            fg='#111827',
            activebackground='#f3f4f6',
            selectcolor='#e2e8f0'
        ).pack(anchor=tk.W)

        results_header = tk.Frame(panel, bg='#f3f4f6')
        results_header.pack(fill=tk.X, padx=12, pady=(6, 0))
        tk.Label(
            results_header,
            text="Analysis Results",
            font=('Segoe UI', 10, 'bold'),
            bg='#f3f4f6',
            fg='#1f2933'
        ).pack(side=tk.LEFT)
        self.results_toggle_btn = tk.Button(
            results_header,
            text="Hide",
            command=self._toggle_results_panel,
            font=('Segoe UI', 8),
            bg='#d1d5db',
            fg='#111827',
            relief=tk.FLAT,
            cursor='hand2',
            width=6
        )
        self.results_toggle_btn.pack(side=tk.RIGHT)

        self.analysis_results_container = tk.Frame(panel, bg='#e5e7eb')
        self.analysis_results_container.pack(fill=tk.BOTH, padx=12, pady=(2, 4))
        result_columns = ('metric', 'value', 'normal')
        self.analysis_results_tree = ttk.Treeview(
            self.analysis_results_container,
            columns=result_columns,
            show='headings',
            height=8,
            selectmode='none'
        )
        self.analysis_results_tree.heading('metric', text='Measurement')
        self.analysis_results_tree.heading('value', text='Value')
        self.analysis_results_tree.heading('normal', text='Reference')
        self.analysis_results_tree.column('metric', width=150, anchor=tk.CENTER)
        self.analysis_results_tree.column('value', width=90, anchor=tk.CENTER)
        self.analysis_results_tree.column('normal', width=120, anchor=tk.CENTER)
        
        # Add both vertical and horizontal scrollbars
        v_scroll = ttk.Scrollbar(
            self.analysis_results_container,
            orient='vertical',
            command=self.analysis_results_tree.yview
        )
        h_scroll = ttk.Scrollbar(
            self.analysis_results_container,
            orient='horizontal',
            command=self.analysis_results_tree.xview
        )
        self.analysis_results_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack with both scrollbars
        self.analysis_results_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        self.analysis_results_container.grid_rowconfigure(0, weight=1)
        self.analysis_results_container.grid_columnconfigure(0, weight=1)
        self.analysis_results_tree.tag_configure('within', background='#ecfdf5')
        self.analysis_results_tree.tag_configure('high', background='#fee2e2')
        self.analysis_results_tree.tag_configure('low', background='#fffbeb')

        self.analysis_summary_label = tk.Label(
            panel,
            text="Run an analysis to view measurements.",
            font=('Segoe UI', 9),
            bg='#f3f4f6',
            fg='#4b5563',
            justify=tk.LEFT,
            wraplength=280
        )
        self.analysis_summary_label.pack(fill=tk.X, padx=12, pady=(0, 6))
        
        btn_frame = tk.Frame(panel, bg='#f3f4f6')
        btn_frame.pack(fill=tk.X, padx=12, pady=(4, 8))
        
        tk.Button(
            btn_frame,
            text="Add Landmark",
            command=self._prompt_add_landmark,
            bg='#2563eb',
            fg='#ffffff',
            relief=tk.FLAT,
            font=('Segoe UI', 9, 'bold'),
            cursor='hand2'
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            btn_frame,
            text="Delete Selected",
            command=self._delete_selected_landmark,
            bg='#dc2626',
            fg='#ffffff',
            relief=tk.FLAT,
            font=('Segoe UI', 9, 'bold'),
            cursor='hand2'
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            btn_frame,
            text="Reset Selected",
            command=self._reset_selected_landmark,
            bg='#6b7280',
            fg='#ffffff',
            relief=tk.FLAT,
            font=('Segoe UI', 9),
            cursor='hand2'
        ).pack(fill=tk.X, pady=2)
        
        tk.Button(
            btn_frame,
            text="Restore All",
            command=self._restore_all_landmarks,
            bg='#059669',
            fg='#ffffff',
            relief=tk.FLAT,
            font=('Segoe UI', 9),
            cursor='hand2'
        ).pack(fill=tk.X, pady=2)
    
    def _refresh_landmark_tree(self):
        if self.landmark_tree is None:
            return
        if self.landmarks is None:
            self.landmark_tree.delete(*self.landmark_tree.get_children())
            return
        self.landmark_tree.tag_configure('ai', background='#f8fafc')
        self.landmark_tree.tag_configure('edited', background='#ecfccb')
        self.landmark_tree.tag_configure('deleted', background='#fee2e2')
        self.landmark_tree.tag_configure('selected', background='#bfdbfe')
        self._suspend_tree_events = True
        self.landmark_tree.delete(*self.landmark_tree.get_children())
        total = min(len(LANDMARK_SHORT), len(self.landmarks))
        for idx in range(total):
            abbr = LANDMARK_SHORT[idx]
            status = 'deleted' if idx in self.suppressed_landmarks else ('edited' if idx in self.landmark_overrides else 'ai')
            tags = [status]
            if self.highlighted_landmark == idx:
                tags.append('selected')
            display_status = 'Removed' if status == 'deleted' else ('Edited' if status == 'edited' else 'AI')
            self.landmark_tree.insert('', 'end', iid=str(idx), values=(abbr, display_status), tags=tags)
        if self.highlighted_landmark is not None and str(self.highlighted_landmark) in self.landmark_tree.get_children():
            self.landmark_tree.selection_set(str(self.highlighted_landmark))
            self.landmark_tree.see(str(self.highlighted_landmark))
        else:
            self.landmark_tree.selection_remove(self.landmark_tree.selection())
        self._suspend_tree_events = False
    
    def _set_tree_hover(self, iid):
        self._landmark_hover_iid = iid
    
    def _on_landmark_tree_hover(self, event):
        if self.landmark_tree is None:
            return
        row = self.landmark_tree.identify_row(event.y)
        if not row:
            return
        try:
            idx = int(row)
        except Exception:
            return
        self._highlight_landmark(idx, from_tree=True, refresh=False)
    
    def _on_landmark_select(self, _event):
        if self._suspend_tree_events:
            return
        idx = self._selected_landmark_index()
        if idx is None:
            return
        self._highlight_landmark(idx, from_tree=True)
    
    def _highlight_landmark(self, index, from_tree=False, refresh=True):
        if index == self.highlighted_landmark:
            if refresh:
                self.refresh_display()
            return
        self.highlighted_landmark = index
        if not from_tree and self.landmark_tree is not None:
            self._suspend_tree_events = True
            try:
                if index is None:
                    self.landmark_tree.selection_remove(self.landmark_tree.selection())
                else:
                    self.landmark_tree.selection_set(str(index))
                    self.landmark_tree.see(str(index))
            finally:
                self._suspend_tree_events = False
        if refresh:
            self.refresh_display()
    
    def _find_nearest_landmark(self, canvas_x, canvas_y, threshold=15):
        if not self._landmark_screen_positions:
            return None
        best_idx = None
        best_dist = float('inf')
        for idx, pos in enumerate(self._landmark_screen_positions):
            if pos is None:
                continue
            px, py = pos
            dist = math.hypot(canvas_x - px, canvas_y - py)
            if dist < best_dist and dist <= threshold:
                best_dist = dist
                best_idx = idx
        return best_idx
    
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        if self.original_image is None or not hasattr(self, 'display_scale') or self.display_scale is None:
            return None
        offset_x, offset_y = getattr(self, '_image_offset', (0, 0))
        img_x = (canvas_x - offset_x) / self.display_scale
        img_y = (canvas_y - offset_y) / self.display_scale
        if self._display_image_size:
            img_w, img_h = self._display_image_size
            img_x = max(0, min(img_w - 1, img_x))
            img_y = max(0, min(img_h - 1, img_y))
        return float(img_x), float(img_y)
    
    def _place_landmark(self, index, canvas_x, canvas_y, recompute=True):
        coords = self._canvas_to_image_coords(canvas_x, canvas_y)
        if coords is None:
            return
        self.landmark_overrides[index] = coords
        self.suppressed_landmarks.discard(index)
        self.pending_landmark_set = None
        self._apply_landmark_overrides(refresh_tree=True, recompute_metrics=recompute)
        self._update_landmark_status_label()
        self._update_export_state()
        self.analysis_summary_label.config(text=f"{len(self.landmark_overrides)} edits applied.")
    
    def _on_canvas_button_press(self, event):
        if self.original_image is None:
            return
        nearest = self._find_nearest_landmark(event.x, event.y)
        if nearest is not None:
            self._highlight_landmark(nearest, refresh=True)
        if not self.landmark_edit_mode.get():
            return
        if self.pending_landmark_set is not None:
            self._place_landmark(self.pending_landmark_set, event.x, event.y)
            return
        if nearest is not None:
            self._dragging_landmark_index = nearest
    
    def _on_canvas_drag(self, event):
        if not self.landmark_edit_mode.get():
            return
        if self._dragging_landmark_index is None:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        self.landmark_overrides[self._dragging_landmark_index] = coords
        self.suppressed_landmarks.discard(self._dragging_landmark_index)
        self._apply_landmark_overrides(refresh_tree=False, recompute_metrics=False)
        self._update_landmark_status_label()
    
    def _on_canvas_button_release(self, _event):
        if self._dragging_landmark_index is None:
            return
        idx = self._dragging_landmark_index
        self._dragging_landmark_index = None
        if idx in self.landmark_overrides:
            self._apply_landmark_overrides(refresh_tree=True, recompute_metrics=True)
    
    def _setup_canvas_bindings(self):
        if self.canvas_bindings_initialized or self.canvas is None:
            return
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)   # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_zoom)   # Linux scroll down
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._on_pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._on_pan)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_button_release)
        self.canvas_bindings_initialized = True
    
    def _on_window_change(self, *_):
        self.brightness_factor = self.brightness_var.get()
        self.contrast_factor = self.contrast_var.get()
        if self.original_image is not None:
            self.refresh_display()
    
    def _reset_image_view(self, refresh=True):
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.invert_colors.set(False)
        self.zoom_scale = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._pan_start = None
        if refresh and self.original_image is not None:
            self.refresh_display()
        elif self.original_image is None:
            self._show_placeholder()
    
    def _reset_landmark_state(self):
        self.predicted_landmarks = None
        self.landmarks = None
        self.landmark_overrides.clear()
        self.suppressed_landmarks.clear()
        self.landmark_insights = []
        self.confidence_scores = None
        self.highlighted_landmark = None
        self.pending_landmark_set = None
        self._dragging_landmark_index = None
        self._landmark_screen_positions = []
        self._set_tree_hover(None)
        self._update_landmark_status_label()
        self._update_export_state()
        if self.landmark_tree is not None:
            for child in self.landmark_tree.get_children():
                self.landmark_tree.delete(child)
        self.measurement_results = {}
        self._update_analysis_results_table()
    
    def _update_landmark_status_label(self):
        if not hasattr(self, 'landmark_status_label') or self.landmark_status_label is None:
            return
        overrides = len(self.landmark_overrides)
        deleted = len(self.suppressed_landmarks)
        parts = []
        if overrides:
            parts.append(f"{overrides} edited")
        if deleted:
            parts.append(f"{deleted} deleted")
        summary = "Manual overrides: " + (", ".join(parts) if parts else "none")
        if self.pending_landmark_set is not None:
            summary += f"\nClick on the image to place {self._get_landmark_name(self.pending_landmark_set)}."
        elif self.landmark_edit_mode.get():
            summary += "\nDrag points or use the tools to adjust landmarks."
        self.landmark_status_label.config(text=summary)

    def _update_export_state(self):
        """Enable report export only when analysis data is available."""
        if not hasattr(self, 'export_btn') or self.export_btn is None:
            return
        has_landmarks = (
            self.landmarks is not None
            and hasattr(self.landmarks, "__len__")
            and np.isfinite(self.landmarks).any()
        )
        state = tk.NORMAL if has_landmarks else tk.DISABLED
        self.export_btn.config(state=state)
    
    def _compose_landmarks(self):
        if self.predicted_landmarks is None:
            return None
        coords = np.array(self.predicted_landmarks, dtype=np.float32).copy()
        for idx in range(coords.shape[0]):
            if idx in self.suppressed_landmarks:
                coords[idx] = np.array([np.nan, np.nan], dtype=np.float32)
        for idx, value in self.landmark_overrides.items():
            coords[idx] = np.array(value, dtype=np.float32)
        return coords
    
    def _apply_landmark_overrides(self, refresh_display=True, recompute_metrics=True, refresh_tree=True):
        composed = self._compose_landmarks()
        self.landmarks = composed
        self.update_landmark_list()
        if refresh_tree:
            self._refresh_landmark_tree()
        if recompute_metrics:
            self.calculate_metrics()
        self.measurement_results = self._compute_measurements()
        self._update_analysis_results_table()
        if refresh_display:
            self.refresh_display()

    def _on_analysis_type_change(self):
        self.measurement_results = self._compute_measurements()
        self._update_analysis_results_table()
        if self.show_tracing.get():
            self.refresh_display()

    def _on_tracing_toggle(self):
        self.refresh_display()

    def _toggle_results_panel(self):
        self.analysis_results_visible = not self.analysis_results_visible
        if self.analysis_results_visible:
            # Pack before the summary label and buttons to maintain order
            self.analysis_results_container.pack_forget()
            self.analysis_results_container.pack(fill=tk.BOTH, padx=12, pady=(2, 4), before=self.analysis_summary_label)
            self.results_toggle_btn.config(text="Hide")
        else:
            self.analysis_results_container.pack_forget()
            self.results_toggle_btn.config(text="Show")

    def _compute_measurements(self):
        if self.landmarks is None:
            return {}
        defs = self.analysis_definitions.get(self.analysis_type.get(), [])
        results = {}
        intermediate = {}
        for definition in defs:
            try:
                if definition["type"] == "angle":
                    pts = [self._get_landmark_point(name) for name in definition["points"]]
                    # Check if any point is None using proper comparison
                    if any(pt is None for pt in pts):
                        value = None
                    else:
                        value = self._calculate_angle(*pts)
                elif definition["type"] == "angle_lines":
                    l1 = [self._get_landmark_point(name) for name in definition["line1"]]
                    l2 = [self._get_landmark_point(name) for name in definition["line2"]]
                    # Check if any point is None using proper comparison
                    if any(pt is None for pt in l1 + l2):
                        value = None
                    else:
                        value = self._calculate_line_angle(l1[0], l1[1], l2[0], l2[1])
                elif definition["type"] == "distance":
                    pts = [self._get_landmark_point(name) for name in definition["points"]]
                    # Check if any point is None using proper comparison
                    if any(pt is None for pt in pts):
                        value = None
                    else:
                        value = self._calculate_distance(*pts)
                elif definition["type"] == "ratio":
                    seg1 = [self._get_landmark_point(name) for name in definition["segments"][0]]
                    seg2 = [self._get_landmark_point(name) for name in definition["segments"][1]]
                    # Check if any point is None using proper comparison
                    if any(pt is None for pt in seg1 + seg2):
                        value = None
                    else:
                        length1 = self._calculate_distance(*seg1)
                        length2 = self._calculate_distance(*seg2)
                        value = (length1 / length2) * 100 if length2 else None
                elif definition["type"] == "difference":
                    a = intermediate.get(definition["a"])
                    b = intermediate.get(definition["b"])
                    value = None if (a is None or b is None) else a - b
                else:
                    value = None
                
                # Format value based on type
                if value is not None:
                    if definition["type"] in {"angle", "angle_lines", "difference"}:
                        value = round(value, 2)
                    elif definition["type"] == "distance":
                        value = round(value * self.current_pixel_spacing, 2)
                    elif definition["type"] == "ratio":
                        value = round(value, 1)
                
                intermediate[definition["name"]] = value
                results[definition["name"]] = {
                    "value": value,
                    "units": definition["units"],
                    "normal": definition["normal"],
                    "range": definition.get("range")
                }
            except Exception as e:
                # Log error for debugging but continue with other measurements
                print(f"Error computing {definition.get('name', 'unknown')}: {e}")
                results[definition["name"]] = {
                    "value": None,
                    "units": definition["units"],
                    "normal": definition["normal"],
                    "range": definition.get("range")
                }
        return results

    def _update_analysis_results_table(self):
        if self.analysis_results_tree is None:
            return
        self.analysis_results_tree.delete(*self.analysis_results_tree.get_children())
        results = self.measurement_results or {}
        if not results:
            self.analysis_summary_label.config(text="Run an analysis to view measurements.")
            return
        within = high = low = 0
        for metric, data in results.items():
            value = data["value"]
            display = "--"
            tag = 'within'
            if value is not None:
                display = self._format_measurement_value(value, data["units"])
                rng = data.get("range")
                if rng:
                    if value < rng[0]:
                        tag = 'low'
                        low += 1
                    elif value > rng[1]:
                        tag = 'high'
                        high += 1
                    else:
                        within += 1
                else:
                    within += 1
            self.analysis_results_tree.insert(
                '', tk.END,
                values=(metric, display, data["normal"]),
                tags=(tag,)
            )
        total = within + high + low
        summary = f"{within}/{total} within range"
        if high or low:
            summary += f" • {high} high, {low} low"
        self.analysis_summary_label.config(text=summary)

    def _format_measurement_value(self, value, units):
        if units == "°":
            return f"{value:.1f}°"
        if units == "mm":
            return f"{value:.1f} mm"
        if units == "%":
            return f"{value:.1f}%"
        return f"{value:.2f}"

    def _get_landmark_point(self, name):
        idx = self.landmark_index_map.get(name)
        if idx is None or self.landmarks is None:
            return None
        if idx >= len(self.landmarks):
            return None
        point = self.landmarks[idx]
        if point is None:
            return None
        # Use np.all to check if all elements are finite, not boolean context
        if not np.all(np.isfinite(point)):
            return None
        return np.array(point, dtype=np.float32)

    def _calculate_distance(self, p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def _calculate_angle(self, a, b, c):
        v1 = a - b
        v2 = c - b
        return self._vector_angle(v1, v2)

    def _calculate_line_angle(self, p1, p2, p3, p4):
        return self._vector_angle(p2 - p1, p4 - p3)

    def _vector_angle(self, v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return None
        cos_theta = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    def _draw_tracing_overlays(self, draw, scale, offset_x, offset_y):
        if self.landmarks is None:
            return
        
        # Define colors for different types of lines
        line_colors = {
            "Cranial Base": "#FF6B6B",
            "Maxillary Plane": "#4ECDC4", 
            "Palatal Plane": "#4ECDC4",
            "Mandibular Plane": "#45B7D1",
            "Facial Axis": "#96CEB4",
            "Frankfort Horizontal": "#FFEAA7",
            "Occlusal Plane": "#DDA0DD",
            "Bisector Occlusal": "#DDA0DD",
            "E-line": "#FFB6C1",
            "Nasolabial": "#FFB6C1",
            "Y-axis": "#98D8C8",
        }
        
        # Default color for any undefined lines
        default_color = "#00e0ff"
        
        # Quality check landmarks before drawing
        problematic_landmarks = self._check_landmark_quality()
        
        for name, start_abbr, end_abbr in self.tracing_segments:
            start = self._get_landmark_point(start_abbr)
            end = self._get_landmark_point(end_abbr)
            if start is None or end is None:
                continue
            # Draw in image space (same as landmark dots) before canvas offset is applied
            x1 = int(start[0] * scale)
            y1 = int(start[1] * scale)
            x2 = int(end[0] * scale)
            y2 = int(end[1] * scale)
            
            color = line_colors.get(name, default_color)
            width = 2 if name in ["Cranial Base", "Maxillary Plane", "Mandibular Plane"] else 1
            
            # Use different line styles based on landmark quality issues only
            if start_abbr in problematic_landmarks or end_abbr in problematic_landmarks:
                # Landmark quality issues - use dashed line
                self._draw_dashed_line(draw, x1, y1, x2, y2, color, width)
            else:
                # Normal - solid line
                draw.line((x1, y1, x2, y2), fill=color, width=width)
            
            # Add label with background for better visibility
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            label_text = name.replace(" ", "\n") if len(name) > 12 else name
            
            # Draw label text with stroke for readability (using default font)
            try:
                # Try to load a smaller font to reduce overlap
                font = ImageFont.truetype("arial.ttf", 10)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Store label positions for canvas drawing after image is placed
            if not hasattr(self, '_tracing_labels'):
                self._tracing_labels = []
            
            # Calculate label positions in screen space (accounting for zoom/pan)
            # Convert from image coordinates to screen coordinates
            screen_x1 = x1 + offset_x
            screen_y1 = y1 + offset_y
            screen_x2 = x2 + offset_x
            screen_y2 = y2 + offset_y
            screen_mid_x = (screen_x1 + screen_x2) // 2
            screen_mid_y = (screen_y1 + screen_y2) // 2
            
            # Adjust label position for Palatal Plane to avoid overlap with Maxillary Plane
            if name == "Palatal Plane":
                # Position Palatal Plane slightly above center
                label_x = screen_mid_x + 30
                label_y = screen_mid_y - 15
            elif name == "Maxillary Plane":
                # Position Maxillary Plane slightly below center
                label_x = screen_mid_x + 30
                label_y = screen_mid_y + 15
            else:
                # Default positioning for other labels
                label_x = screen_mid_x + 4
                label_y = screen_mid_y
            
            self._tracing_labels.append((label_x, label_y, label_text, color))
            
            # Don't draw text on image - we'll draw on canvas later
    
    def _draw_dashed_line(self, draw, x1, y1, x2, y2, color, width, dash_length=5):
        """Draw a dashed line between two points"""
        import math
        distance = math.hypot(x2 - x1, y2 - y1)
        dashes = int(distance / (dash_length * 2))
        for i in range(dashes):
            start_dash = i * 2 * dash_length
            end_dash = min(start_dash + dash_length, distance)
            if start_dash >= distance:
                break
            
            # Calculate dash endpoints
            ratio1 = start_dash / distance
            ratio2 = end_dash / distance
            
            dash_x1 = x1 + (x2 - x1) * ratio1
            dash_y1 = y1 + (y2 - y1) * ratio1
            dash_x2 = x1 + (x2 - x1) * ratio2
            dash_y2 = y1 + (y2 - y1) * ratio2
            
            draw.line((dash_x1, dash_y1, dash_x2, dash_y2), fill=color, width=width)
    
    def _check_landmark_quality(self):
        """Check for potentially misdetected landmarks based on anatomical constraints"""
        if self.landmarks is None:
            return set()
        
        problematic = set()
        
        # Check for landmarks that are too far from expected anatomical regions
        # This is a simplified quality check based on relative positions
        
        # Get key reference points
        sella = self._get_landmark_point("S")
        nasion = self._get_landmark_point("N")
        menton = self._get_landmark_point("Me")
        
        if sella is None or nasion is None or menton is None:
            return problematic
        
        # Calculate basic face dimensions for normalization
        face_height = np.linalg.norm(np.array(nasion) - np.array(menton))
        face_width = face_height * 0.8  # Approximate face width
        
        # Check individual landmarks
        checks = [
            # ("Landmark", "Expected Y range relative to Nasion-Menton", "Expected X range")
            ("Or", (0.1, 0.3), (-0.3, 0.3)),      # Orbitale should be upper face
            ("Po", (0.1, 0.3), (-0.4, -0.1)),     # Porion should be upper face, lateral
            ("ANS", (0.3, 0.5), (-0.2, 0.2)),      # Anterior nasal spine mid-face
            ("PNS", (0.3, 0.5), (-0.2, 0.2)),      # Posterior nasal spine mid-face
            ("A", (0.4, 0.6), (-0.2, 0.2)),        # A-point mid-face
            ("B", (0.7, 0.9), (-0.2, 0.2)),        # B-point lower face
            ("Pog", (0.8, 1.0), (-0.2, 0.2)),       # Pogonion chin area
            ("Go", (0.6, 0.8), (-0.4, -0.2)),       # Gonion lower jaw angle
            ("Co", (0.2, 0.4), (-0.4, -0.2)),       # Condylion jaw joint area
            ("Prn", (0.3, 0.5), (-0.1, 0.1)),       # Pronasale nose tip
            ("Sn", (0.35, 0.55), (-0.1, 0.1)),     # Subnasale under nose
            ("Ls", (0.5, 0.7), (-0.1, 0.1)),       # Labrale superius upper lip
            ("Li", (0.6, 0.8), (-0.1, 0.1)),       # Labrale inferius lower lip
        ]
        
        for landmark, (y_min, y_max), (x_min, x_max) in checks:
            point = self._get_landmark_point(landmark)
            if point is None:
                continue
            
            # Normalize coordinates relative to face dimensions
            y_norm = (point[1] - nasion[1]) / face_height
            x_norm = (point[0] - nasion[0]) / face_width
            
            # Check if landmark is outside expected range
            if not (y_min <= y_norm <= y_max and x_min <= x_norm <= x_max):
                problematic.add(landmark)
        
        # Additional geometric checks
        # Frankfort Horizontal should be relatively horizontal
        porion = self._get_landmark_point("Po")
        orbitale = self._get_landmark_point("Or")
        if porion is not None and orbitale is not None:
            frankfort_angle = abs(math.degrees(math.atan2(
                orbitale[1] - porion[1], 
                orbitale[0] - porion[0]
            )))
            if frankfort_angle > 30:  # Too steep, likely misdetected
                problematic.update(["Po", "Or"])
        
        # Check for anatomically impossible configurations
        pogonion = self._get_landmark_point("Pog")
        b_point = self._get_landmark_point("B")
        if pogonion is not None and b_point is not None:
            # Pogonion should be inferior to B-point
            if pogonion[1] < b_point[1]:
                problematic.update(["Pog", "B"])
        
        return problematic

    def _toggle_landmark_edit_mode(self):
        if self.landmark_edit_mode.get():
            self.canvas.config(cursor='tcross')
        else:
            self.pending_landmark_set = None
            self._dragging_landmark_index = None
            self.canvas.config(cursor='')
        self._update_landmark_status_label()
    
    def _selected_landmark_index(self):
        if self.landmark_tree is None:
            return None
        selected = self.landmark_tree.selection()
        if not selected:
            return None
        try:
            return int(selected[0])
        except Exception:
            return None
    
    def _prompt_add_landmark(self):
        if self.landmarks is None:
            messagebox.showinfo("Add Landmark", "Run an analysis first.")
            return
        idx = simpledialog.askinteger(
            "Add Landmark",
            "Enter landmark number (1-29) to place:\n"
            "(e.g., 1=A-point, 11=Sella)",
            parent=self.root,
            minvalue=1,
            maxvalue=len(LANDMARK_NAMES)
        )
        if idx is None:
            return
        self.pending_landmark_set = idx - 1
        self.landmark_edit_mode.set(True)
        self._update_landmark_status_label()
        messagebox.showinfo(
            "Placement Ready",
            f"Click on the X-ray to place {self._get_landmark_name(self.pending_landmark_set)}."
        )
    
    def _delete_selected_landmark(self):
        idx = self._selected_landmark_index()
        if idx is None:
            messagebox.showinfo("Delete Landmark", "Select a landmark first.")
            return
        self.suppressed_landmarks.add(idx)
        self.landmark_overrides.pop(idx, None)
        self._apply_landmark_overrides()
        self._update_landmark_status_label()
    
    def _reset_selected_landmark(self):
        idx = self._selected_landmark_index()
        if idx is None:
            messagebox.showinfo("Reset Landmark", "Select a landmark first.")
            return
        changed = False
        if idx in self.landmark_overrides:
            self.landmark_overrides.pop(idx)
            changed = True
        if idx in self.suppressed_landmarks:
            self.suppressed_landmarks.discard(idx)
            changed = True
        if changed:
            self._apply_landmark_overrides()
            self._update_landmark_status_label()
    
    def _restore_all_landmarks(self):
        if not self.landmark_overrides and not self.suppressed_landmarks:
            return
        self.landmark_overrides.clear()
        self.suppressed_landmarks.clear()
        self.pending_landmark_set = None
        self._apply_landmark_overrides()
        self._update_landmark_status_label()
    
    def _get_confidence_value(self, index):
        if self.confidence_scores is None:
            return None
        data = self.confidence_scores
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().float().numpy()
        arr = np.array(data).reshape(-1)
        if index >= len(arr):
            return None
        val = arr[index]
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[0]
        try:
            val = float(val)
        except Exception:
            return None
        return max(0.0, min(1.0, val))
    
    def _get_landmark_name(self, index):
        if index < len(LANDMARK_SHORT):
            return f"{LANDMARK_SHORT[index]}"
        return f"L{index+1}"
    
    def _show_placeholder(self):
        if self.canvas is None:
            return
        if self.placeholder_text_id is None:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            self.placeholder_text_id = self.canvas.create_text(
                width / 2,
                height / 2,
                text="Load an X-ray to begin.",
                font=('Segoe UI', 16),
                fill='#9ca3af',
                tags='placeholder',
                anchor='center'
            )
        self.canvas.itemconfigure(self.placeholder_text_id, state='normal')
        self._on_canvas_resize()
    
    def _hide_placeholder(self):
        if self.placeholder_text_id is not None:
            self.canvas.itemconfigure(self.placeholder_text_id, state='hidden')
    
    def _apply_image_adjustments(self, image):
        img = image.convert('RGB')
        if abs(self.brightness_factor - 1.0) > 1e-3:
            img = ImageEnhance.Brightness(img).enhance(self.brightness_factor)
        if abs(self.contrast_factor - 1.0) > 1e-3:
            img = ImageEnhance.Contrast(img).enhance(self.contrast_factor)
        if self.invert_colors.get():
            img = ImageOps.invert(img)
        return img
    
    def _on_zoom(self, event):
        if self.original_image is None:
            return "break"
        if hasattr(event, 'delta') and event.delta != 0:
            direction = 1 if event.delta > 0 else -1
        else:
            # Linux scroll events
            if getattr(event, 'num', None) == 4:
                direction = 1
            elif getattr(event, 'num', None) == 5:
                direction = -1
            else:
                return "break"
        factor = 1.1 if direction > 0 else 0.9
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_scale * factor))
        if abs(new_zoom - self.zoom_scale) < 1e-3:
            return "break"
        self.zoom_scale = new_zoom
        self.refresh_display()
        return "break"
    
    def _start_pan(self, event):
        if self.original_image is None:
            return
        self._pan_start = (event.x, event.y)
        self.canvas.config(cursor='fleur')
    
    def _on_pan(self, event):
        if self._pan_start is None or self.original_image is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self._pan_start = (event.x, event.y)
        self._clamp_pan()
        self.refresh_display()
    
    def _end_pan(self, _event):
        self._pan_start = None
        if self.canvas is not None:
            self.canvas.config(cursor='')
        
    def _clamp_pan(self, canvas_w=None, canvas_h=None, image_w=None, image_h=None):
        if canvas_w is None or canvas_h is None or image_w is None or image_h is None:
            if not self._image_draw_params:
                return
            canvas_w = self._image_draw_params['canvas_width']
            canvas_h = self._image_draw_params['canvas_height']
            image_w = self._image_draw_params['image_width']
            image_h = self._image_draw_params['image_height']
        max_offset_x = max(canvas_w, image_w)
        max_offset_y = max(canvas_h, image_h)
        self.pan_offset_x = max(-max_offset_x, min(max_offset_x, self.pan_offset_x))
        self.pan_offset_y = max(-max_offset_y, min(max_offset_y, self.pan_offset_y))
    
    def upload_image(self):
        """Upload PNG or DICOM image file"""
        filetypes = [
            ("Cephalometric images", "*.png *.dcm *.dicom"),
            ("PNG files", "*.png"),
            ("DICOM files", "*.dcm *.dicom"),
            ("All image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.dcm *.dicom"),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(title="Select Cephalometric Image", filetypes=filetypes)
        
        if filepath:
            self._reset_image_view(refresh=False)
            self._reset_landmark_state()
            self.image_path = filepath
            image_id = os.path.splitext(os.path.basename(filepath))[0]

            # For DICOM files, try to extract pixel spacing from metadata
            if is_dicom_file(filepath):
                dicom_spacing = extract_dicom_pixel_spacing(filepath)
                if dicom_spacing is not None:
                    self.current_pixel_spacing = dicom_spacing
                    print(f"DICOM pixel spacing: {dicom_spacing:.4f} mm/px")
                else:
                    self._set_current_pixel_spacing(image_id)
            else:
                self._set_current_pixel_spacing(image_id)

            self.load_and_display_image(filepath)
            self.analyze_btn.config(state=tk.NORMAL)
            self.gt_landmarks = None
            self.gt_status.config(text="GT: Not available (custom image)")
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.landmarks = None
            self.landmark_insights = []
            self.analysis_result_label.config(text="No analysis yet.")
            self.gt_hint_label.config(text="Ground truth: not loaded")
            self.summary_label.config(text="Run an analysis to view landmark accuracy.")
            
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
            self._reset_image_view(refresh=False)
            self._reset_landmark_state()
            self.image_path = filepath
            self.load_and_display_image(filepath)
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Load ground truth
            self.load_ground_truth(filepath)
            
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.landmarks = None
            self.landmark_insights = []
            self.analysis_result_label.config(text="No analysis yet.")
            self.gt_hint_label.config(text="Ground truth: not loaded")
            self.summary_label.config(text="Run an analysis to view landmark accuracy.")

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
        """Load and display the image (supports PNG, JPEG, DICOM, etc.)"""
        try:
            if is_dicom_file(filepath):
                if not PYDICOM_AVAILABLE:
                    messagebox.showerror(
                        "Missing Dependency",
                        "pydicom is required to load DICOM files.\n"
                        "Install it with: pip install pydicom"
                    )
                    return
                self.original_image = load_dicom_as_pil(filepath)
            else:
                self.original_image = Image.open(filepath)
                if self.original_image.mode != 'RGB':
                    self.original_image = self.original_image.convert('RGB')
            self._display_image_size = self.original_image.size
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
        """Display image with landmarks and applied adjustments"""
        if self.canvas is None or image is None:
            self._show_placeholder()
            return
        
        self.canvas.update_idletasks()
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        
        adjusted = self._apply_image_adjustments(image)
        img_width, img_height = adjusted.size
        base_scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
        scale = max(1e-3, base_scale * self.zoom_scale)
        
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        
        self._image_draw_params = {
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'image_width': new_width,
            'image_height': new_height
        }
        self._clamp_pan(canvas_width, canvas_height, new_width, new_height)
        offset_x = (canvas_width - new_width) / 2 + self.pan_offset_x
        offset_y = (canvas_height - new_height) / 2 + self.pan_offset_y
        
        display_img = adjusted.resize((new_width, new_height), Image.Resampling.LANCZOS)
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
                    short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
                    label = short_name
                    
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y + 8 if i in (16, 18) else y - 15
                    
                    for pos_x, pos_y, label_w, label_h in label_positions:
                        if (abs(label_x - pos_x) < label_w and abs(label_y - pos_y) < label_h):
                            label_y = pos_y - label_h - 2
                    
                    label_height = 10
                    label_positions.append((label_x, label_y, label_width, label_height))
                    draw.text((label_x, label_y), label, fill='#ff6600')
                else:
                    label = str(i+1)
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y - 15
                    
                    for pos_x, pos_y, label_w, label_h in label_positions:
                        if (abs(label_x - pos_x) < label_w and abs(label_y - pos_y) < label_h):
                            label_y = pos_y - label_h - 2
                    
                    label_height = 10
                    label_positions.append((label_x, label_y, label_width, label_height))
                    draw.text((label_x, label_y), label, fill='#ff6600')
        else:
            label_positions = []
        
        # Draw predictions (circles)
        self._landmark_screen_positions = []
        if self.show_predictions.get() and landmarks is not None:
            colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00',
                     '#CCFF00', '#88FF00', '#44FF00', '#00FF00', '#00FF44',
                     '#00FF88', '#00FFCC', '#00FFFF', '#00CCFF', '#0088FF',
                     '#0044FF', '#0000FF', '#4400FF', '#8800FF', '#CC00FF',
                     '#FF00FF', '#FF00CC', '#FF0088', '#FF0044', '#FF0000',
                     '#00FF88', '#FFCC00', '#00CCFF', '#FF00CC']
            
            pred_label_positions = []
            for i, lm in enumerate(landmarks):
                if lm is None or not np.all(np.isfinite(lm)):
                    self._landmark_screen_positions.append(None)
                    continue
                x = int(lm[0] * scale)
                y = int(lm[1] * scale)
                self._landmark_screen_positions.append((x + offset_x, y + offset_y))
                color = colors[i % len(colors)]
                radius = 5
                highlight = (self.highlighted_landmark == i)
                if highlight:
                    draw.ellipse([x-9, y-9, x+9, y+9], outline='#ffffff', width=2)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                           fill=color, outline='#111111', width=1)
                
                if self.show_labels.get():
                    short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
                    label = short_name
                    
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y + 8 if i in (16, 18) else y - 15
                    
                    all_positions = label_positions + pred_label_positions
                    for pos_x, pos_y, label_w, label_h in all_positions:
                        if (abs(label_x - pos_x) < label_w and abs(label_y - pos_y) < label_h):
                            label_y = pos_y - label_h - 2
                    
                    label_height = 10
                    pred_label_positions.append((label_x, label_y, label_width, label_height))
                    draw.text((label_x, label_y), label, fill=color)
                else:
                    label = str(i+1)
                    label_width = len(label) * 6
                    label_x = x - label_width // 2
                    label_y = y - 15
                    
                    all_positions = label_positions + pred_label_positions
                    for pos_x, pos_y, label_w, label_h in all_positions:
                        if (abs(label_x - pos_x) < label_w and abs(label_y - pos_y) < label_h):
                            label_y = pos_y - label_h - 2
                    
                    label_height = 10
                    pred_label_positions.append((label_x, label_y, label_width, label_height))
                    draw.text((label_x, label_y), label, fill=color)
        else:
            self._landmark_screen_positions = [None] * len(LANDMARK_SHORT)
        
        if self.show_tracing.get():
            self._draw_tracing_overlays(draw, scale, offset_x, offset_y)
        
        self.photo = ImageTk.PhotoImage(display_img)
        self._hide_placeholder()
        self.canvas.delete('image')
        self.canvas_image_id = self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo, tags='image')
        self.display_scale = scale
        self._image_offset = (offset_x, offset_y)
        
        # Draw tracing labels on canvas (not on image) to fix zoom/pan positioning
        if hasattr(self, '_tracing_labels'):
            self.canvas.delete('tracing_label')
            for label_x, label_y, label_text, color in self._tracing_labels:
                self.canvas.create_text(
                    label_x, label_y,
                    text=label_text,
                    fill=color,
                    font=('Arial', 10, 'bold'),
                    tags='tracing_label',
                    anchor='w'
                )
            # Clear stored labels for next redraw
            self._tracing_labels = []
        
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
    
    def extract_landmark_coordinates(self, heatmaps, orig_w, orig_h):
        """Convert model heatmaps into pixel coordinates in the original image space."""
        method = self.coord_method.get()
        if method == 'soft_argmax':
            coords = soft_argmax_2d(heatmaps, temperature=0.05)
        else:
            coords = extract_coordinates_from_heatmaps(heatmaps, method=method)
        
        coords = coords.cpu().numpy()
        if coords.ndim == 3:
            coords = coords[0]
        
        scale_x = orig_w / self.config.HEATMAP_SIZE[1]
        scale_y = orig_h / self.config.HEATMAP_SIZE[0]
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        return coords
    
    def analyze_image(self):
        """Run model inference"""
        if self.original_image is None:
            return
        try:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Analyzing...")
            input_tensor, (orig_w, orig_h) = self.preprocess_image(self.original_image)
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            heatmaps = None
            confidences = None
            if isinstance(outputs, dict):
                heatmaps = outputs.get('heatmaps')
                confidences = outputs.get('confidences')
            elif isinstance(outputs, (tuple, list)):
                if len(outputs) > 0:
                    heatmaps = outputs[0]
                if len(outputs) > 1:
                    confidences = outputs[1]
            else:
                heatmaps = outputs
            
            if heatmaps is None:
                raise ValueError("Model output did not include heatmaps")
            
            coords = self.extract_landmark_coordinates(heatmaps, orig_w, orig_h)
            self.predicted_landmarks = coords.copy()
            self.landmark_overrides.clear()
            self.suppressed_landmarks.clear()
            if confidences is not None:
                try:
                    self.confidence_scores = (
                        confidences.detach().to('cpu').float().numpy()
                        if isinstance(confidences, torch.Tensor)
                        else np.array(confidences, dtype=np.float32)
                    )
                except Exception:
                    self.confidence_scores = None
            else:
                self.confidence_scores = None
            self.pending_landmark_set = None
            self._update_landmark_status_label()
            self._apply_landmark_overrides(refresh_display=False, recompute_metrics=False)
            self.calculate_metrics()
            self.update_landmark_list()
            self.measurement_results = self._compute_measurements()
            self._update_analysis_results_table()
            self.refresh_display()
            self._update_export_state()
            if hasattr(self, 'export_btn'):
                self.export_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_metrics(self):
        """Calculate accuracy metrics when ground truth is available."""
        if self.landmarks is None:
            return
        if self.gt_landmarks is None:
            self.analysis_result_label.config(
                text="Prediction ready.\nGround truth not available for accuracy metrics."
            )
            self.gt_hint_label.config(text="Ground truth: unavailable")
            return

        pixel_spacing = float(self.current_pixel_spacing)
        distances = np.sqrt(np.sum((self.landmarks - self.gt_landmarks) ** 2, axis=1))
        distances_mm = distances * pixel_spacing

        mre = np.mean(distances_mm)
        std = np.std(distances_mm)
        sdr_2mm = (distances_mm <= 2.0).mean() * 100
        sdr_25mm = (distances_mm <= 2.5).mean() * 100
        sdr_3mm = (distances_mm <= 3.0).mean() * 100
        sdr_4mm = (distances_mm <= 4.0).mean() * 100

        self.analysis_result_label.config(
            text=(
                f"MRE: {mre:.2f} ± {std:.2f} mm\n"
                f"SDR: 2mm={sdr_2mm:.1f}% | 2.5={sdr_25mm:.1f}% | "
                f"3={sdr_3mm:.1f}% | 4={sdr_4mm:.1f}%"
            )
        )
        self.gt_hint_label.config(text="Ground truth: loaded")
        
    def update_landmark_list(self):
        """Update landmark summary for the dashboard"""
        if self.landmarks is None:
            self.summary_label.config(text="Run an analysis to view landmark accuracy.")
            self.landmark_insights = []
            return
        
        pixel_spacing = float(self.current_pixel_spacing)
        report_data = []
        within_2 = within_4 = 0
        max_error = 0.0
        
        if self.gt_landmarks is not None:
            for i, (pred, gt) in enumerate(zip(self.landmarks, self.gt_landmarks)):
                short_name = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i+1}"
                err = np.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2) * pixel_spacing
                max_error = max(max_error, err)
                if err <= 2.0:
                    within_2 += 1
                if err <= 4.0:
                    within_4 += 1
                report_data.append({
                    "index": i + 1,
                    "name": short_name,
                    "pred": pred,
                    "gt": gt,
                    "error": err
                })
            
            self.landmark_insights = report_data
            total = len(report_data)
            summary = (
                f"{within_2}/{total} landmarks within 2 mm\n"
                f"{within_4}/{total} landmarks within 4 mm (max error {max_error:.2f} mm)"
            )
            self.summary_label.config(text=summary)
        else:
            self.summary_label.config(text="Ground truth unavailable for this image.")
            self.landmark_insights = []

    def show_landmark_report(self):
        """Display a detailed report of Pred vs GT landmarks"""
        if self.landmarks is None:
            messagebox.showinfo("Landmark Report", "Run an analysis first.")
            return
        
        if self.gt_landmarks is None:
            messagebox.showinfo("Landmark Report", "Ground truth landmarks are required for this report.")
            return
        
        if not self.landmark_insights:
            self.update_landmark_list()
        
        report_lines = []
        header = f"{'No.':>3} {'Name':<5} {'Pred (x,y)':>14} {'GT (x,y)':>14} {'Error(mm)':>10} Status"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for info in self.landmark_insights:
            pred = info["pred"]
            gt = info["gt"]
            err = info["error"]
            status = "✓ within 2mm" if err <= 2 else ("~ within 4mm" if err <= 4 else "✗ >4mm")
            report_lines.append(
                f"{info['index']:>3} {info['name']:<5} ({pred[0]:6.1f},{pred[1]:6.1f}) "
                f"({gt[0]:6.1f},{gt[1]:6.1f}) {err:10.2f} {status}"
            )
        
        report_text = "\n".join(report_lines)
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Landmarks (Pred vs GT) Report")
        report_window.geometry("800x500")
        report_window.configure(bg='#2b2b2b')
        
        text_widget = tk.Text(report_window, font=('Consolas', 10), bg='#1e1e1e', fg='#e0e0e0')
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, report_text)
        text_widget.config(state=tk.DISABLED)
        
        close_btn = tk.Button(report_window, text="Close", command=report_window.destroy, bg='#5a5a5a', fg='white')
        close_btn.pack(pady=(0, 10))


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TScale', background='#3c3c3c', troughcolor='#1e1e1e')
    
    app = CephalometricGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
