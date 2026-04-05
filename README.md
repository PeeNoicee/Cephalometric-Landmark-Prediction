# Cephalometric Landmark Detection & AI Diagnosis

A web application that automatically detects 29 cephalometric landmarks from lateral skull X-rays and generates an AI-powered clinical diagnosis. Built as a special problem research project, it combines a high-accuracy HRNet deep learning model with a rule-based diagnosis engine backed by three orthodontic textbooks (Steiner, Ricketts, McNamara).

---

## What It Does

You upload a lateral cephalometric radiograph (JPG, PNG, or DICOM). The app:
1. Runs the image through an HRNet model to locate 29 anatomical landmarks
2. Computes Steiner, Ricketts, and McNamara cephalometric analyses automatically
3. Generates a structured AI diagnosis — skeletal classification, key findings, age-adjusted Ricketts norms, and a clinical summary
4. Lets you adjust landmarks manually and export a PDF report

The diagnosis engine is **fully deterministic** (rule-based + Qwen 2.5 14B LLM for polishing the summary). It cross-checks analyses against each other and flags suspect landmarks using heatmap confidence scores.

---

## Model Performance

| Metric | This Model | Khan et al. (2025) |
|--------|-----------|-------------------|
| **MRE** | **0.764 ± 0.659 mm** | 1.69 ± 3.36 mm |
| **SDR@2mm** | **95.33%** | 81.18% |
| **SDR@2.5mm** | **97.75%** | 87.28% |
| **SDR@3mm** | **98.94%** | 90.82% |
| **SDR@4mm** | **99.54%** | 94.82% |

---

## Features

- **29-landmark detection** from a single lateral cephalogram
- **Three analyses**: Steiner, Ricketts, McNamara — calculated and classified automatically
- **AI diagnosis** with report confidence level, key findings, age-adjusted Ricketts norm tables, and clinical summary
- **Landmark confidence indicators** — low-confidence landmarks are visually flagged with a dashed ring
- **Manual landmark editing** — drag any landmark to correct its position
- **DICOM support** — extracts pixel spacing automatically for accurate mm measurements
- **PDF export** — full report including the annotated image, measurements table, and AI diagnosis
- **Mobile-friendly** — pinch to zoom, drag to pan, touch landmark editing

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **CPU** | Intel i5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | — | NVIDIA RTX 3060+ (CUDA 12.x) |
| **Storage** | 5 GB free | 10 GB free |

---

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (optional but strongly recommended)

### 1. Clone the repo

```bash
git clone https://github.com/PeeNoicee/Cephalometric-Landmark-Detection.git
cd Cephalometric-Landmark-Detection
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Add the trained model

Place `best_model.pth` in the `checkpoints/` folder:
```
checkpoints/
  best_model.pth
```

### 4. Install Ollama (for AI diagnosis)

Download from [https://ollama.ai/download](https://ollama.ai/download), then pull the model:
```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

### 5. Build and run

```bash
# Install frontend dependencies
cd web
npm install
npm run build
cd ..

# Start the server
python api_server.py
```

Open **http://localhost:8000** in your browser.

For frontend development with hot-reload, run `npm run dev` inside `web/` alongside `python api_server.py`.

---

## Sharing Over the Internet (ngrok)

ngrok creates a public HTTPS tunnel to your local server — no cloud hosting needed.

### One-time setup

1. Download ngrok from [https://ngrok.com/download](https://ngrok.com/download)
2. Sign up at [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup) and get your authtoken
3. Register it:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

### Start the full stack

Run each of these in a separate terminal:

**Terminal 1 — backend:**
```bash
python api_server.py
```

**Terminal 2 — frontend dev server:**
```bash
cd web
npm run dev
```

**Terminal 3 — ngrok tunnel:**
```bash
ngrok http 5173
```

The ngrok output shows your public URL:
```
Forwarding    https://abcd-1234.ngrok-free.app -> http://localhost:5173
```

Share that URL with anyone — it works immediately with no firewall changes.

> ⚠️ The URL changes every time you restart ngrok (free tier). The tunnel is only active while your computer is running.

### Quick-start batch script (Windows)

Save this as `start.bat` in the project root:
```batch
@echo off
start "Backend"  cmd /k "python api_server.py"
timeout /t 3 /nobreak > nul
start "Frontend" cmd /k "cd web && npm run dev"
timeout /t 5 /nobreak > nul
start "Ngrok"    cmd /k "ngrok http 5173"
```

### ngrok limits (free tier)

- URL changes on each restart
- 1 GB/day bandwidth
- 40 requests/minute
- Single-threaded model inference — users queue up naturally

---

## How to Use

1. **Upload** a lateral cephalogram (JPG, PNG, or DICOM) — drag and drop or click **Select**
2. **Analyze** — click **Analyze** to run landmark detection
3. **Review** — inspect landmarks on the canvas, switch between Steiner / Ricketts / McNamara tabs
4. **Edit** — enable edit mode to drag any misplaced landmark
5. **Diagnose** — click **Generate Diagnosis** for the full AI report
6. **Export** — click **Save as PDF** for a printable report

---

## Architecture

### Backend (`api_server.py`)

- **FastAPI** web server serving the React frontend and REST API
- **HRNet** model (PyTorch) for heatmap-based landmark detection
- **DICOM handling** via pydicom — extracts pixel spacing for mm-accurate measurements

### Diagnosis engine (`rag/retriever.py`)

- **Rule-based layer** — deterministic skeletal classification, key findings, Ricketts age-adjusted norm tables, cross-analysis contradiction detection
- **LLM layer** — Qwen 2.5 14B via Ollama polishes the clinical summary; temperature=0 for reproducible output
- **RAG layer** — ChromaDB vector store with 244 chunks from three orthodontic textbooks (Steiner, Ricketts, McNamara)
- **Confidence scoring** — heatmap peak values flag unreliable landmarks before they affect the diagnosis

### Frontend (`web/src/`)

- **React 18** + Vite + Tailwind CSS
- **Canvas-based** landmark rendering with zoom/pan/drag
- **PDF export** via jsPDF with custom markdown-to-PDF renderer
- **ReactMarkdown + remark-gfm** for rendering GFM tables in the diagnosis panel

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server status and model info |
| `POST` | `/api/predict` | Landmark detection — accepts `multipart/form-data` with `file` |
| `POST` | `/api/diagnose` | AI diagnosis — accepts JSON `{ measurements, landmark_confidences }` |

### Quick example

```python
import requests

with open("cephalogram.jpg", "rb") as f:
    result = requests.post("http://localhost:8000/api/predict", files={"file": f}).json()

diagnosis = requests.post("http://localhost:8000/api/diagnose", json={
    "measurements": result["measurements"],
    "landmark_confidences": result["landmark_confidences"],
}).json()

print(diagnosis["diagnosis"])
```

---

## Detected Landmarks

| # | Short | Full Name | # | Short | Full Name |
|---|-------|-----------|---|-------|-----------|
| 1 | A | A-point | 16 | Po | Porion |
| 2 | ANS | Anterior Nasal Spine | 17 | L5 | Lower 2nd PM Cusp Tip |
| 3 | B | B-point | 18 | L1 | Lower Incisor Tip |
| 4 | Me | Menton | 19 | L6 | Lower Molar Cusp Tip |
| 5 | N | Nasion | 20 | U5 | Upper 2nd PM Cusp Tip |
| 6 | Or | Orbitale | 21 | U1A | Upper Incisor Apex |
| 7 | Pog | Pogonion | 22 | U1 | Upper Incisor Tip |
| 8 | PNS | Posterior Nasal Spine | 23 | U6 | Upper Molar Cusp Tip |
| 9 | Prn | Pronasale | 24 | L1A | Lower Incisor Apex |
| 10 | R | Ramus | 25 | Li | Labrale inferius |
| 11 | S | Sella | 26 | Ls | Labrale superius |
| 12 | Ar | Articulare | 27 | N' | Soft Tissue Nasion |
| 13 | Co | Condylion | 28 | Pog' | Soft Tissue Pogonion |
| 14 | Gn | Gnathion | 29 | Sn | Subnasale |
| 15 | Go | Gonion | | | |

---

## Troubleshooting

**Model not loading**
- Make sure `checkpoints/best_model.pth` exists
- Run `python -c "import torch; print(torch.cuda.is_available())"` to verify GPU availability

**CUDA not detected**
- Reinstall PyTorch with the correct CUDA version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

**Ollama not responding**
- Make sure Ollama is running: `ollama serve`
- Confirm the model is downloaded: `ollama list`

**ngrok `ERR_NGROK_8012`**
- The frontend dev server probably stopped — restart it with `cd web && npm run dev`
- Then restart ngrok: `ngrok http 5173`

**Frontend build errors**
- Delete `web/node_modules` and reinstall: `cd web && npm install`
- Confirm Node.js version is 18+: `node --version`

---

## Citation

```bibtex
@article{khalid2025benchmark,
  title={A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification},
  author={Khalid, Muhammad Anwaar and others},
  journal={Scientific Data},
  volume={12},
  pages={1336},
  year={2025},
  publisher={Nature Publishing Group}
}
```
