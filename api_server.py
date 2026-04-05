"""
FastAPI Server for Cephalometric Landmark Detection

Exposes the HRNet model as a REST API and serves the web frontend.

Usage:
    pip install fastapi uvicorn python-multipart
    python api_server.py

Endpoints:
    POST /api/predict  — Upload a cephalometric image, returns 29 landmark coordinates.
    GET  /api/health   — Health check / model status.
"""
import os
import sys
import io
import time
import base64
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.config.config import TrainConfig
from src.models.hrnet import HRNet
from src.utils.metrics import extract_coordinates_from_heatmaps
from src.utils.dicom_handler import is_dicom_file, load_dicom_as_pil, extract_dicom_pixel_spacing

# ---------------------------------------------------------------------------
# Landmark data (same order as training / GUI)
# ---------------------------------------------------------------------------
LANDMARK_NAMES = [
    "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion",
    "Orbitale", "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus",
    "Sella", "Articulare", "Condylion", "Gnathion", "Gonion",
    "Porion", "Lower 2nd PM Cusp Tip", "Lower Incisor Tip", "Lower Molar Cusp Tip",
    "Upper 2nd PM Cusp Tip", "Upper Incisor Apex", "Upper Incisor Tip",
    "Upper Molar Cusp Tip", "Lower Incisor Apex", "Labrale inferius",
    "Labrale superius", "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale",
]

LANDMARK_SHORT = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Prn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "L5", "L1", "L6",
    "U5", "U1A", "U1", "U6", "L1A", "Li", "Ls", "N'", "Pog'", "Sn",
]

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
config = TrainConfig()
model: HRNet | None = None
device: torch.device = torch.device("cpu")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cephalometric Landmark Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HRNet(
        num_landmarks=config.NUM_LANDMARKS,
        use_multitask=config.USE_MULTITASK,
    )

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Place best_model.pth in the checkpoints/ folder."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch")
    best_mre = checkpoint.get("best_mre")
    print(f"Model loaded from {checkpoint_path}")
    if epoch is not None:
        print(f"  Epoch: {int(epoch) + 1}")
    if best_mre is not None:
        print(f"  Best MRE: {best_mre:.2f} mm")


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors GUI/_preprocess_image exactly)
# ---------------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image):
    img_np = np.array(pil_image)

    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    orig_h, orig_w = img_np.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np)

    img_resized = cv2.resize(img_np, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

    return img_tensor, (orig_w, orig_h)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(pil_image: Image.Image) -> dict:
    if model is None:
        raise RuntimeError("Model is not loaded")

    t0 = time.time()
    img_tensor, (orig_w, orig_h) = preprocess_image(pil_image)

    with torch.no_grad():
        outputs = model(img_tensor)

    if isinstance(outputs, (tuple, list)):
        heatmaps = outputs[0]
        cvm_logits = outputs[1] if len(outputs) > 1 else None
    else:
        heatmaps = outputs
        cvm_logits = None

    coords = extract_coordinates_from_heatmaps(heatmaps, method="argmax")

    # Per-landmark confidence: peak heatmap value, normalized 0-1 relative to the
    # most-confident landmark in this image (best = 1.0, worst = 0.0).
    heatmaps_np = heatmaps.cpu().numpy()[0]  # (NUM_LANDMARKS, H, W)
    peak_values = heatmaps_np.max(axis=(1, 2))  # (NUM_LANDMARKS,)
    p_max = float(peak_values.max())
    conf_scores = (peak_values / p_max) if p_max > 0 else np.ones(len(peak_values))

    scale_x = orig_w / config.HEATMAP_SIZE[1]
    scale_y = orig_h / config.HEATMAP_SIZE[0]

    coords = coords.cpu().numpy()[0]
    coords[:, 0] *= scale_x
    coords[:, 1] *= scale_y

    elapsed = time.time() - t0

    landmarks = []
    for i in range(config.NUM_LANDMARKS):
        landmarks.append({
            "index": i,
            "name": LANDMARK_NAMES[i],
            "short": LANDMARK_SHORT[i],
            "x": round(float(coords[i, 0]), 2),
            "y": round(float(coords[i, 1]), 2),
            "confidence": round(float(conf_scores[i]), 4),
        })

    result = {
        "landmarks": landmarks,
        "image_size": {"width": orig_w, "height": orig_h},
        "inference_time_seconds": round(elapsed, 3),
    }

    if cvm_logits is not None:
        cvm_pred = int(torch.argmax(cvm_logits, dim=1).item()) + 1
        result["cvm_stage"] = cvm_pred

    return result


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "num_landmarks": config.NUM_LANDMARKS,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or ""

    try:
        is_dcm = filename.lower().endswith((".dcm", ".dicom"))
        if not is_dcm and len(contents) > 132 and contents[128:132] == b"DICM":
            is_dcm = True

        if is_dcm:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                pil_image = load_dicom_as_pil(tmp_path)
                pixel_spacing = extract_dicom_pixel_spacing(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            pixel_spacing = None

        result = run_inference(pil_image)

        # Return the image the model actually analyzed as base64 PNG
        # so the frontend can display it in sync with the landmarks
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        result["image_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")

        if pixel_spacing is not None:
            result["pixel_spacing_mm"] = round(pixel_spacing, 6)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# RAG Diagnosis endpoint
# ---------------------------------------------------------------------------
@app.post("/api/diagnose")
async def diagnose(payload: dict):
    """Generate a clinical diagnosis from cephalometric measurements using RAG + Qwen 2.5 14B."""
    try:
        from rag.retriever import generate_diagnosis
        measurements = payload.get("measurements", [])
        landmark_confidences = payload.get("landmark_confidences", {})
        if not measurements:
            raise HTTPException(status_code=400, detail="No measurements provided")
        diagnosis = generate_diagnosis(measurements, landmark_confidences)
        return {"diagnosis": diagnosis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Serve frontend (production)
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "web", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    load_model()


if __name__ == "__main__":
    print("=" * 60)
    print("  Cephalometric Landmark Detection - Web Server")
    print("=" * 60)
    print()
    print("Starting server on http://0.0.0.0:8000")
    print("API docs:  http://localhost:8000/docs")
    print("Frontend:  http://localhost:5173 (dev) or http://localhost:8000 (prod)")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
