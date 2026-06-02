"""
HTTP REST API — Human Body Measurement System
=============================================
Exposes the measurement pipeline as a production-ready REST API.

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil

from measurement_api import MeasurementAPI

app = FastAPI(
    title="Body Measurement API",
    version="1.0.0",
    description=(
        "Extract 31 anthropometric body measurements from 3D scans (96.3% accuracy), "
        "photographs, or video files (81.5% accuracy)."
    ),
)

# Initialise once at startup — loads calibration factors into memory
api = MeasurementAPI()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe for load balancers and uptime monitors."""
    return {"status": "ok", "version": "1.0.0"}


# ── Path B: 3D Scan ───────────────────────────────────────────────────────────

@app.post("/measure/scan", tags=["Measurements"])
async def measure_from_scan(
    file: UploadFile = File(..., description="A _smpld.json file from a 3D body scanner"),
    apply_calibration: bool = Form(True, description="Apply trained calibration factors (recommended)"),
):
    """
    Extract 31 body measurements from a 3D scan file.

    - **Accuracy:** 96.3% (LOO-CV validated on 36 subjects)
    - **GPU required:** No
    - **Response time:** < 200 ms
    - **File format:** `_smpld.json` produced by a professional body scanner
    """
    if not file.filename.endswith("_smpld.json"):
        raise HTTPException(status_code=400, detail="File must end with _smpld.json")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_smpld(tmp_path, apply_calibration=apply_calibration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)


# ── Path A: Photo ─────────────────────────────────────────────────────────────

@app.post("/measure/photo", tags=["Measurements"])
async def measure_from_photo(
    file: UploadFile = File(..., description="A JPG or PNG photograph of the person"),
    height_cm: float = Form(..., description="Subject's real height in centimetres"),
):
    """
    Extract body measurements from a photograph using the HMR deep learning model.

    - **Accuracy:** ~81.5%
    - **GPU recommended:** Yes (5–15 s without GPU, < 2 s with GPU)
    - **File format:** JPG or PNG
    - **Requirement:** HMR model files must be present in the `models/` directory
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="File must be a JPG or PNG image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_image(tmp_path, height_cm=height_cm)
        # Strip large numpy arrays — not JSON-serialisable and not needed by callers
        result.pop("vertices", None)
        result.pop("joints", None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)


# ── Path C: Video ─────────────────────────────────────────────────────────────

@app.post("/measure/video", tags=["Measurements"])
async def measure_from_video(
    file: UploadFile = File(..., description="A video file (MP4, AVI, MOV, MKV, WEBM, M4V)"),
    height_cm: float = Form(..., description="Subject's real height in centimetres"),
    frame_index: int = Form(None, description="0-based frame number to analyse. Defaults to middle frame."),
):
    """
    Extract body measurements from a video file.

    A single frame is extracted from the video and processed through the same
    HMR pipeline as the photo endpoint.

    - **Accuracy:** ~81.5%
    - **GPU recommended:** Yes
    - **Supported formats:** MP4, AVI, MOV, MKV, WEBM, M4V
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: {', '.join(sorted(allowed))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = api.measure_from_video(tmp_path, height_cm=height_cm, frame_index=frame_index)
        result.pop("vertices", None)
        result.pop("joints", None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)
