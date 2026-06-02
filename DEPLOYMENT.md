# Deployment Guide — Human Body Measurement System

## Table of Contents
1. [System Overview](#1-system-overview)
2. [How to Run the Code](#2-how-to-run-the-code)
3. [Server Configuration Requirements](#3-server-configuration-requirements)
4. [GPU Acceleration](#4-gpu-acceleration)
5. [Production Deployment](#5-production-deployment)
6. [External API Integration](#6-external-api-integration)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. System Overview

The system has **two independent measurement pathways**. Choose based on what data you have:

| Pathway | Input | Accuracy | Dependencies |
|---------|-------|----------|--------------|
| **Path B** — 3D Scan | `_smpld.json` from 3D body scanner | **96.3%** | Python + NumPy only |
| **Path A** — Photo/Video | JPG/PNG image or MP4 video | ~81.5% | TensorFlow 1.13 + GPU recommended |

**Recommendation:** Use Path B if your clients have a 3D body scanner (much more accurate, no GPU needed, no heavy ML framework). Use Path A only when a 3D scanner is unavailable.

---

## 2. How to Run the Code

### 2.1 Prerequisites — Install Once

**Step 1: Clone the repository**
```bash
git clone https://github.com/kamranalishahdataai-create/human-body-measurement.git
cd human-body-measurement
```

**Step 2: Create a Python virtual environment**
```bash
# Python 3.7 is required (TF 1.13 does not support Python 3.8+)
python3.7 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows note:** `opendr` and `mayavi` (used for 3D rendering only) may fail on Windows.
> If you only need measurements (not 3D visualization), you can skip them:
> ```bash
> pip install tensorflow==1.13.1 pillow opencv-python scikit-image numpy==1.16.1 scipy==1.2.1
> ```

**Step 4: Download pre-trained models (required for Path A — image/video only)**
```bash
# Downloads the HMR model checkpoint (~300 MB)
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
tar -xf models.tar.gz
# Place extracted files into the models/ folder
```

The `models/` folder must contain:
```
models/
  model.ckpt-667589.index
  model.ckpt-667589.meta
  model.ckpt-667589.data-00000-of-00001
  neutral_smpl_with_cocoplus_reg.pkl
```

---

### 2.2 Running the Web Demo (Streamlit UI)

```bash
streamlit run demo_app.py
```

Opens at `http://localhost:8501`. Three input modes are available in the sidebar:

| Mode | What you provide |
|------|-----------------|
| Sample Subject | Pick from pre-loaded test subjects |
| Upload `_smpld.json` | Upload a 3D scan file |
| Upload Video | Upload a video + enter height in cm |

To run on a specific port or allow external access:
```bash
streamlit run demo_app.py --server.port 8080 --server.address 0.0.0.0
```

---

### 2.3 Running via Command Line

**From a 3D scan file (Path B — recommended):**
```bash
python measurement_api.py path/to/subject_smpld.json
```

**From a photo (Path A):**
```bash
python measurement_api.py photo.jpg --height 170
```

**From a video (Path A):**
```bash
# Uses the middle frame by default
python measurement_api.py video.mp4 --height 170

# Specify a frame number (0-based)
python measurement_api.py video.mp4 --height 170 --frame 90
```

**From a subject directory (auto-detect):**
```bash
python measurement_api.py path/to/subject_folder/
```

**Batch process an entire data directory:**
```bash
python measurement_api.py --batch path/to/data_directory/
```

All commands save a `_measurements.json` file alongside the input.

---

## 3. Server Configuration Requirements

### 3.1 Path B — 3D Scan Input (Recommended)

This pathway is pure Python/NumPy — no TensorFlow, no GPU.

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any dual-core | 4-core, 2.5 GHz+ |
| RAM | 512 MB | 2 GB |
| Disk | 500 MB | 2 GB |
| OS | Linux, macOS, Windows | Ubuntu 20.04 LTS |
| Python | 3.7 – 3.11 | 3.9 |

Processing time per subject: **< 100 ms**

### 3.2 Path A — Image / Video Input

This pathway requires TensorFlow 1.13 and loads a ResNet model (~300 MB) into memory.

| Component | Minimum (CPU-only) | Recommended (GPU) |
|-----------|-------------------|-------------------|
| CPU | 4-core, 2.5 GHz | 8-core, 3 GHz+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| GPU | Not required | NVIDIA GTX 1060 6 GB or better |
| VRAM | — | 4 GB minimum |
| OS | Ubuntu 18.04 / 20.04 | Ubuntu 20.04 LTS |
| Python | **3.7 exactly** | 3.7 |
| CUDA | — | 10.0 |
| cuDNN | — | 7.4 |

Processing time per image:
- CPU-only: 5 – 15 seconds
- With GPU: 0.5 – 2 seconds

### 3.3 API Server (FastAPI/Flask wrapper)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core | 8-core |
| RAM | 8 GB | 32 GB |
| Disk | 10 GB | SSD 50 GB |
| Network | 100 Mbps | 1 Gbps |
| Concurrent requests | 1 (single TF session) | 1 per GPU |

---

## 4. GPU Acceleration

### Does this project require a GPU?

**No — but it depends on which pathway you use:**

- **Path B (3D scan):** No GPU needed at all. Runs fast on any CPU.
- **Path A (image/video):** GPU is strongly recommended for production use. Without a GPU, each inference takes 5–15 seconds. With a GPU, it takes under 2 seconds.

### GPU Setup for Path A

TensorFlow 1.13 requires **CUDA 10.0** and **cuDNN 7.4** specifically.

```bash
# Verify your NVIDIA driver supports CUDA 10.0
nvidia-smi   # Driver version must be >= 410.48

# Install CUDA 10.0 (Ubuntu)
# Follow: https://developer.nvidia.com/cuda-10.0-download-archive

# Install cuDNN 7.4 for CUDA 10.0
# Download from: https://developer.nvidia.com/rdp/cudnn-archive
# Then:
sudo dpkg -i libcudnn7_7.4.x.x-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.x.x-1+cuda10.0_amd64.deb

# Install TensorFlow with GPU support
pip install tensorflow-gpu==1.13.1
```

**Verify GPU is detected:**
```python
import tensorflow as tf
print(tf.test.is_gpu_available())   # Should print True
```

### Running on CPU only

TensorFlow will automatically use CPU if no compatible GPU is found. No code change needed. To force CPU explicitly:

```bash
CUDA_VISIBLE_DEVICES="" python measurement_api.py photo.jpg --height 170
```

---

## 5. Production Deployment

### 5.1 Docker Deployment (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.7-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (Path B only — no TensorFlow)
COPY requirements_pathb.txt .
RUN pip install --no-cache-dir fastapi uvicorn opencv-python-headless numpy scipy pillow

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t body-measurement-api .
docker run -p 8000:8000 body-measurement-api
```

**With GPU support** (requires nvidia-container-toolkit):
```bash
docker run --gpus all -p 8000:8000 body-measurement-api
```

### 5.2 Systemd Service (Linux)

Create `/etc/systemd/system/body-measurement.service`:

```ini
[Unit]
Description=Body Measurement API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/body-measurement
ExecStart=/opt/body-measurement/.venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable body-measurement
sudo systemctl start body-measurement
```

### 5.3 Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;   # allow large video uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;    # allow time for inference
    }
}
```

---

## 6. External API Integration

### 6.1 FastAPI Server — `api_server.py`

Create this file in the project root to expose the measurement system as an HTTP REST API:

```python
"""
HTTP REST API for the Body Measurement System.
Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, shutil

from measurement_api import MeasurementAPI

app = FastAPI(
    title="Body Measurement API",
    version="1.0.0",
    description="Extract 31 body measurements from 3D scans or photos.",
)

# Load API once at startup (caches calibration factors)
api = MeasurementAPI()


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Path B: 3D Scan ──────────────────────────────────────────────────────────

@app.post("/measure/scan")
async def measure_from_scan(
    file: UploadFile = File(..., description="A _smpld.json 3D scan file"),
    apply_calibration: bool = Form(True),
):
    """
    Extract body measurements from a 3D body scan file (_smpld.json).

    - Accuracy: 96.3%
    - No GPU required
    - Response time: < 200 ms
    """
    if not file.filename.endswith("_smpld.json"):
        raise HTTPException(400, "File must be a _smpld.json file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_smpld(tmp_path, apply_calibration=apply_calibration)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result)


# ── Path A: Photo ─────────────────────────────────────────────────────────────

@app.post("/measure/photo")
async def measure_from_photo(
    file: UploadFile = File(..., description="A JPG or PNG photograph"),
    height_cm: float = Form(..., description="Subject's real height in cm"),
):
    """
    Extract body measurements from a photograph using the HMR deep learning model.

    - Accuracy: ~81.5%
    - GPU recommended (5–15 s without GPU, < 2 s with GPU)
    - Requires HMR model files in the models/ directory
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, "File must be a JPG or PNG image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_image(tmp_path, height_cm=height_cm)
        result.pop("vertices", None)   # strip large numpy array from response
        result.pop("joints", None)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result)


# ── Path C: Video ─────────────────────────────────────────────────────────────

@app.post("/measure/video")
async def measure_from_video(
    file: UploadFile = File(..., description="A video file (MP4, AVI, MOV, etc.)"),
    height_cm: float = Form(..., description="Subject's real height in cm"),
    frame_index: int = Form(None, description="Frame number to analyse (default: middle frame)"),
):
    """
    Extract body measurements from a video file by selecting a single frame.

    - Accuracy: ~81.5% (same as photo path)
    - GPU recommended for faster inference
    - Supported formats: MP4, AVI, MOV, MKV, WEBM, M4V
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported video format. Allowed: {', '.join(allowed)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_video(tmp_path, height_cm=height_cm, frame_index=frame_index)
        result.pop("vertices", None)
        result.pop("joints", None)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result)
```

Install and run the API server:

```bash
pip install fastapi uvicorn python-multipart
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs open automatically at: `http://your-server:8000/docs`

---

### 6.2 API Endpoints Reference

#### `GET /health`
Returns server status. Use for load-balancer health checks.

```json
{ "status": "ok", "version": "1.0.0" }
```

---

#### `POST /measure/scan` — From 3D Scan (Recommended)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | multipart file | Yes | `_smpld.json` from 3D body scanner |
| `apply_calibration` | bool | No (default: true) | Apply trained correction factors |

**Example — cURL:**
```bash
curl -X POST http://your-server:8000/measure/scan \
  -F "file=@subject001_smpld.json" \
  -F "apply_calibration=true"
```

**Example — Python:**
```python
import requests

with open("subject001_smpld.json", "rb") as f:
    response = requests.post(
        "http://your-server:8000/measure/scan",
        files={"file": ("subject001_smpld.json", f, "application/json")},
        data={"apply_calibration": "true"},
    )

data = response.json()
print(data["measurements"]["chest circumference"]["value_cm"])  # e.g. 92.4
```

**Response:**
```json
{
  "method": "smpld_3d_model",
  "accuracy": "96.3% (calibrated, LOO-CV)",
  "measurement_count": 31,
  "measurements": {
    "height":              { "value_cm": 167.9, "raw_cm": 167.8, "chinese": "模型身高" },
    "chest circumference": { "value_cm": 92.4,  "raw_cm": 93.1,  "chinese": "胸围" },
    "waist circumference": { "value_cm": 76.2,  "raw_cm": 77.0,  "chinese": "中腰围" },
    "hip circumference":   { "value_cm": 97.5,  "raw_cm": 98.0,  "chinese": "下臀围" }
    // ... 27 more measurements
  },
  "landmarks": { ... },
  "joints": [ ... ]
}
```

---

#### `POST /measure/photo` — From Photo

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | multipart file | Yes | JPG or PNG photo of person |
| `height_cm` | float | Yes | Subject's real height in cm |

**Example — cURL:**
```bash
curl -X POST http://your-server:8000/measure/photo \
  -F "file=@person.jpg" \
  -F "height_cm=170"
```

**Response:**
```json
{
  "method": "hmr_image",
  "accuracy": "81.5% (calibrated)",
  "measurement_count": 31,
  "measurements": { ... }
}
```

---

#### `POST /measure/video` — From Video

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | multipart file | Yes | MP4/AVI/MOV/MKV/WEBM/M4V video |
| `height_cm` | float | Yes | Subject's real height in cm |
| `frame_index` | int | No | Frame to analyse (default: middle frame) |

**Example — cURL:**
```bash
curl -X POST http://your-server:8000/measure/video \
  -F "file=@recording.mp4" \
  -F "height_cm=175" \
  -F "frame_index=120"
```

---

### 6.3 Response Fields

All `/measure/*` endpoints return the same structure:

| Field | Type | Description |
|-------|------|-------------|
| `method` | string | `smpld_3d_model`, `hmr_image`, or `hmr_video` |
| `accuracy` | string | Validated accuracy for this pathway |
| `measurement_count` | int | Number of measurements returned (up to 31) |
| `measurements` | object | All measurements (see below) |
| `landmarks` | object | 160 3D body landmark positions (Path B only) |
| `joints` | array | 45 joint positions (Path B only) |
| `frame_index` | int | Frame used (video only) |
| `total_frames` | int | Total frames in video (video only) |
| `fps` | float | Video frame rate (video only) |

Each entry in `measurements`:

| Field | Type | Description |
|-------|------|-------------|
| `value_cm` | float | Final measurement in centimetres (1 decimal) |
| `raw_cm` | float | Raw (uncalibrated) measurement |
| `chinese` | string | Chinese measurement name |

### 6.4 All 31 Measurement Keys

```
height                    back shoulder breadth      front shoulder breadth
neck circumference        front chest breadth        back chest breadth
neck width                sleeve length              left sleeve length
chest circumference       bicep left circumference   bicep right circumference
waist circumference       wrist left circumference   front waist length
mid waist circumference   back waist length          back mid waist length
pants waist circumference back clothing length       back waist height
hip circumference         thigh left circumference   thigh right circumference
knee circumference        calf left circumference    calf right circumference
pant height               leg height                 O to under hip
open crotch
```

---

### 6.5 Error Responses

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Wrong file type, missing required field |
| 500 | Inference failed (model not loaded, corrupt file, etc.) |

```json
{ "detail": "height_cm is required for video-based measurement." }
```

---

## 7. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Cannot find inference graph in tar archive` | DeepLab model not downloaded | Let `inference.py` auto-download on first run |
| `Fix path to models/` at startup | HMR checkpoint missing | Download from the link in the README |
| `No module named tensorflow` | TF not installed | `pip install tensorflow==1.13.1` |
| `Module cv2 not found` | OpenCV not installed | `pip install opencv-python` |
| `RuntimeError: CUDA error` | Wrong CUDA/cuDNN version | Must use CUDA 10.0 + cuDNN 7.4 with TF 1.13 |
| Slow inference (15+ s) | Running on CPU | Install GPU drivers and `tensorflow-gpu==1.13.1` |
| `No measurements found in _smpld.json` | Wrong file format | File must contain a `"Measurements"` key |
| Port 8501 already in use | Another Streamlit instance | `streamlit run demo_app.py --server.port 8080` |

---

*Generated for deployment on client infrastructure — Human Body Measurement System v1.0*
