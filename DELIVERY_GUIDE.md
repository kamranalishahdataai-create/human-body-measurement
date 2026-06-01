# Body Measurement AI — Delivery Package

**Delivered to:** Client
**Date:** May 2026
**Version:** 1.0

---

## 📦 What's Included

| File / Folder | Purpose |
|---|---|
| `measurement_api.py` | **Main API class** — `MeasurementAPI` for integrating into your software |
| `measure_body.py` | CLI tool — extract measurements from a single file or folder |
| `extract_measurements.py` | Core measurement extraction algorithm |
| `evaluate_option_b.py` | Evaluation script (LOO-CV across 36 subjects, 96.3% accuracy) |
| `demo_app.py` | Streamlit web demo for visual presentations |
| `demo.py` / `inference.py` | Photo-based pipeline (HMR + DeepLab segmentation) |
| `networks.py`, `utils.py` | Supporting modules |
| `option_b_results.json` | **Trained calibration factors** (required for 96.3% accuracy) |
| `models/` | Pre-trained HMR + SMPL model files (~410MB) |
| `data/` | Custom body measurement points definitions |
| `src/` | HMR/SMPL source modules |
| `requirements.txt` | Python dependencies |
| `ACCURACY_REPORT.md` | Detailed accuracy report |
| `VIDEO_NARRATION_SCRIPT.md` | Demo walkthrough script |

---

## 🎯 Two Measurement Pathways

| Pathway | Input | Accuracy | Use Case |
|---|---|---|---|
| **Path A — Photo** | Single 2D image + height | **81.5%** | Mobile app / quick estimate |
| **Path B — 3D Scan** ⭐ | `_smpld.json` from your 3D body scanner | **96.3%** | Production / tailoring |

**Recommendation:** Use Path B in production for accuracy.

---

## ⚙️ Setup Instructions

### 1. Requirements
- Python **3.7 – 3.10** (recommended: 3.9)
- Windows, Linux, or macOS
- ~2GB disk space (with models)
- No GPU required for Path B (3D Scan)
- GPU recommended for Path A (Photo) — NVIDIA T4 or equivalent

### 2. Install

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python measure_body.py --help
```

---

## 🚀 How to Use

### Option 1: Python API (recommended for integration)

```python
from measurement_api import MeasurementAPI

api = MeasurementAPI()

# Path B — 3D Scan (96.3% accuracy)
result = api.measure_from_smpld("path/to/subject_smpld.json")

# Path A — Photo (81.5% accuracy)
result = api.measure_from_image("path/to/photo.jpg", height_cm=170)

# Auto-detect from folder (prefers 3D scan)
result = api.measure_from_directory("path/to/subject_folder")

# Read results
for name, data in result['measurements'].items():
    print(f"{name}: {data['value_cm']} cm")
```

### Option 2: CLI Tool

```bash
# Single file
python measure_body.py path/to/subject_smpld.json

# Batch process a folder
python measure_body.py --batch path/to/data_folder/
```

### Option 3: Web Demo (for presentations)

```bash
streamlit run demo_app.py
```
Open your browser at **http://localhost:8501**

---

## 📊 Output Format

```json
{
  "method": "smpld_3d_model",
  "accuracy": "96.3% (calibrated, LOO-CV)",
  "measurement_count": 31,
  "measurements": {
    "chest_circumference": {
      "value_cm": 96.2,
      "raw_cm": 95.4,
      "chinese": "胸围"
    },
    "waist_circumference": {
      "value_cm": 82.1,
      "raw_cm": 81.5,
      "chinese": "中腰围"
    }
    // ... 31 measurements total
  }
}
```

### 31 Measurements Extracted

**Core dimensions:** height, shoulder breadth (front/back), sleeve length

**Circumferences:** neck, chest, bicep (L/R), wrist, waist, mid-waist, pants-waist, hip, thigh (L/R), knee, calf (L/R)

**Lengths & widths:** neck width, chest breadth (front/back), waist length (front/back/mid), back clothing length, back waist height, pant height, leg height, crotch

---

## 🔧 Production Deployment

For deploying as a REST API server, see the FastAPI wrapper (deliverable on request). Quick steps:

1. Wrap `MeasurementAPI` in a FastAPI endpoint
2. Build a Docker container
3. Deploy to AWS EC2 (`t3.medium` is sufficient for Path B)

**Estimated cost:** ~$30/month on AWS for Path B (no GPU needed)

---

## ⚠️ Important Notes

### Calibration File
- `option_b_results.json` contains the **calibration factors** trained on 36 subjects via Leave-One-Out Cross-Validation
- This file is **required** for the 96.3% accuracy
- Do not delete or modify it

### Model Files
The `models/` folder contains:
- `model.ckpt-667589.*` — HMR pretrained weights (Path A)
- `neutral_smpl_with_cocoplus_reg.pkl` — SMPL body model
- These are only needed for **Path A (photo-based)**. Path B works without them.

### Licensing
- **Project code:** MIT License (free for commercial use)
- **HMR model:** MIT License (free for commercial use)
- **DeepLab:** Apache 2.0 (free for commercial use)
- **SMPL body model:** ⚠️ Requires a commercial license from Max Planck Institute for production use. Contact: https://smpl.is.tue.mpg.de

---

## 📞 Support

For questions, issues, or new features, contact me through Upwork.

For deployment support, FastAPI wrapper, or REST API integration — these are available as additional milestones.

---

## ✅ Validation

Tested and validated on **36 subjects** with ground-truth tape-measure values:
- **Mean Absolute Error:** 1.25 cm (calibrated)
- **Within 2cm tolerance:** 89% of measurements
- **Overall accuracy:** 96.34%

See `ACCURACY_REPORT.md` for the full breakdown.

---

**Thank you for the project!**
