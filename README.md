# Cephalometric Landmark Detection & AI Diagnosis Web App

A web application for automatic cephalometric landmark detection and AI-powered clinical diagnosis using HRNet neural networks and RAG (Retrieval-Augmented Generation) with Ollama.

## Features

- **Automatic Landmark Detection**: AI-powered detection of 29 cephalometric landmarks from lateral cephalograms
- **Clinical Analysis**: Automatic calculation of Steiner, Ricketts, and McNamara cephalometric analyses
- **AI Diagnosis**: RAG-powered clinical diagnosis based on orthodontic textbooks (Steiner, Ricketts, McNamara)
- **Web Interface**: Modern React frontend with drag-and-drop image upload
- **PDF Export**: Professional reports with measurements and AI diagnosis
- **DICOM Support**: Automatic handling of DICOM files with pixel spacing extraction

## Performance

| Metric | This Model | Khan et al. (2025) |
|--------|-----------|-------------------|
| **MRE** | **0.764 ± 0.659 mm** | 1.69 ± 3.36 mm |
| **SDR@2mm** | **95.33%** | 81.18% |
| **SDR@2.5mm** | **97.75%** | 87.28% |
| **SDR@3mm** | **98.94%** | 90.82% |
| **SDR@4mm** | **99.54%** | 94.82% |

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**
- **CUDA-compatible GPU** (recommended for inference speed)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/PeeNoicee/Cephalometric-Landmark-Detection.git
cd Cephalometric-Landmark-Detection

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download the Trained Model

The application requires a pre-trained HRNet model. You'll need to obtain `best_model.pth` and place it in the `checkpoints/` folder:

```bash
mkdir -p checkpoints
# Place best_model.pth in checkpoints/ directory
```

### 3. Install Ollama for AI Diagnosis (Optional)

For AI-powered clinical diagnosis:

```bash
# Install Ollama (if not already installed)
# Download from: https://ollama.ai/download

# Pull the required model
ollama pull qwen2.5:14b-instruct-q4_K_M

# Verify installation
ollama list
```

### 4. Setup Frontend

```bash
# Navigate to web directory
cd web

# Install Node.js dependencies
npm install

# Build the frontend
npm run build
```

### 5. Run the Application

```bash
# From the root directory, run the web server
python api_server.py
```

The application will start on:
- **Frontend**: http://localhost:5173 (development) or http://localhost:8000 (production)
- **API**: http://localhost:8000/docs (FastAPI documentation)

## Usage

### Basic Workflow

1. **Upload Image**: Drag and drop a lateral cephalometric radiograph (JPG/PNG) or DICOM file
2. **Landmark Detection**: The AI will automatically detect 29 cephalometric landmarks
3. **View Analysis**: Review the calculated measurements for Steiner, Ricketts, and McNamara analyses
4. **Generate Diagnosis**: Click "Generate Diagnosis" for AI-powered clinical interpretation
5. **Export Report**: Download a professional PDF report with measurements and diagnosis

### Supported File Formats

- **Images**: JPEG, PNG
- **DICOM**: Automatic pixel spacing extraction for accurate measurements
- **PDF Export**: Professional reports with all findings

### Landmark Detection

The system detects 29 cephalometric landmarks:

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
| 15 | Go | Gonion |

## API Reference

### Endpoints

- `GET /api/health` - Health check and system status
- `POST /api/predict` - Landmark detection from image
- `POST /api/diagnose` - AI clinical diagnosis from measurements

### Example API Usage

```python
import requests

# Upload image for landmark detection
with open('cephalogram.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/api/predict', files={'file': f})
    result = response.json()

# Generate diagnosis from measurements
measurements = result['measurements']  # From predict endpoint
diag_response = requests.post('http://localhost:8000/api/diagnose', json={'measurements': measurements})
diagnosis = diag_response.json()
```

## System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB
- **Storage**: 5GB free space
- **OS**: Windows 10+, macOS 12+, Ubuntu 20.04+

### Recommended Requirements
- **GPU**: NVIDIA RTX 3060 or better with CUDA 12.8+
- **RAM**: 16GB+
- **CPU**: Intel i7 or AMD Ryzen 7+

### Dependencies

#### Python (requirements.txt)
- PyTorch 2.11+ (CUDA version for GPU acceleration)
- FastAPI for web API
- OpenCV, Pillow for image processing
- ChromaDB for vector database
- Ollama for LLM integration
- pdfplumber, pytesseract for PDF text extraction

#### Node.js (web/package.json)
- React 18+ for frontend
- Vite for build tooling
- react-markdown for diagnosis rendering

## Troubleshooting

### Common Issues

**"Model not loaded" error**
- Ensure `best_model.pth` is in the `checkpoints/` directory
- Check that the model file is not corrupted

**CUDA/GPU not detected**
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Ollama connection failed**
- Ensure Ollama is running: `ollama serve`
- Verify model is pulled: `ollama pull qwen2.5:14b-instruct-q4_K_M`

**Frontend build fails**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 18+)

### Performance Tips

- **GPU Acceleration**: Use CUDA PyTorch for 5-10x faster inference
- **Memory**: Close other GPU-intensive applications
- **Image Size**: Optimal input size is 864x768 pixels

## Architecture

### Backend (Python/FastAPI)
- **Model**: HRNet for landmark detection
- **RAG System**: ChromaDB + Ollama Qwen 2.5 14B for diagnosis
- **Image Processing**: OpenCV + Pillow for preprocessing
- **DICOM Support**: pydicom for medical image handling

### Frontend (React)
- **UI**: Modern drag-and-drop interface
- **Visualization**: Canvas-based landmark display
- **Export**: jsPDF for PDF generation with markdown support

### AI Diagnosis System
- **Knowledge Base**: 244 chunks from 3 orthodontic textbooks
- **Embeddings**: nomic-embed-text for semantic search
- **LLM**: Qwen 2.5 14B with deterministic temperature=0
- **Rules**: Textbook-verified clinical interpretation guidelines

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with descriptive messages: `git commit -m "Add feature description"`
5. Push to your branch: `git push origin feature-name`
6. Create a Pull Request

## License

See [LICENSE](LICENSE) file.

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
