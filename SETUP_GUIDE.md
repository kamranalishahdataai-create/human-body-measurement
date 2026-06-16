# Body Measurement AI — Complete Setup & Run Guide

> **For clients running this system on their own machine.**
> Follow every step in order. Do not skip any step.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [System Requirements](#2-system-requirements)
3. [Step-by-Step Installation (Windows)](#3-step-by-step-installation-windows)
4. [Apply Required Compatibility Patches](#4-apply-required-compatibility-patches)
5. [Running the Web Interface](#5-running-the-web-interface)
6. [How to Use the App](#6-how-to-use-the-app)
7. [Running the REST API Server (Optional)](#7-running-the-rest-api-server-optional)
8. [Troubleshooting Common Errors](#8-troubleshooting-common-errors)
9. [File Structure Reference](#9-file-structure-reference)

---

## 1. What This System Does

This system uses AI and Computer Vision to automatically extract **31 body measurements** (chest, waist, hip, shoulder width, etc.) from:

| Input Type | How it works | Accuracy |
|---|---|---|
| **3D Scan file** (`_smpld.json`) | Reads a pre-made 3D body scan | ~96.3% |
| **Photo** (JPG/PNG) | AI removes background, estimates 3D body shape | ~81.5% |
| **Video** (MP4/AVI/MOV) | You pick a frame, same AI pipeline as photo | ~81.5% |

Results are displayed in a browser-based web interface with charts, a 3D model viewer, and export to JSON/CSV.

---

## 2. System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 64-bit | Windows 10/11 64-bit |
| **RAM** | 8 GB | 16 GB |
| **Disk space** | 10 GB free | 20 GB free |
| **Python** | 3.11 | 3.11 or 3.12 |
| **CPU** | Any modern 64-bit | Intel Core i5 / AMD Ryzen 5 or better |
| **GPU** | Not required | NVIDIA GPU (optional, speeds up photo/video) |

> **Important:** Python **3.11** or **3.12** is strongly recommended.
> Python 3.13 requires extra patches (documented in Section 4).
> Python 3.9 or 3.10 may work but is untested.

### Required Software to Install First

1. **Python 3.11** — Download from: https://www.python.org/downloads/release/python-3119/
   - During install: **check "Add Python to PATH"**
   - Choose "Customize installation" → make sure pip is included

2. **Git** (optional, for cloning) — https://git-scm.com/download/win

3. **Microsoft Visual C++ Redistributable** — usually already installed on Windows 10/11.
   If you get DLL errors, download from Microsoft's official site.

---

## 3. Step-by-Step Installation (Windows)

### Step 3.1 — Get the project files

Either clone from GitHub (if you have Git installed):
```
git clone https://github.com/kamranalishahdataai-create/human-body-measurement.git C:\body-ai
```

Or download the ZIP from GitHub and extract it to:
```
C:\body-ai\
```

> **Why a short path?** Windows has a 260-character path length limit. TensorFlow's installation files have very long internal paths. Installing into a deeply nested folder (like `C:\Users\YourName\Documents\Projects\...`) will likely cause `pip install tensorflow` to fail. Always use a short top-level path like `C:\body-ai`.

### Step 3.2 — Create a Python virtual environment at a short path

Open **Command Prompt** (press `Win + R`, type `cmd`, press Enter):

```cmd
python -m venv C:\tbm
```

This creates a clean Python environment at `C:\tbm`. All packages will be installed here.

### Step 3.3 — Activate the virtual environment

```cmd
C:\tbm\Scripts\activate
```

Your prompt should now show `(tbm)` at the start. **Every command from here must be run with this environment active.**

### Step 3.4 — Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Step 3.5 — Install all required packages

Run these commands one by one. If one fails, see Section 8 (Troubleshooting).

```cmd
pip install tensorflow==2.15.0
```

> If tensorflow 2.15 is not available for your Python version, try `tensorflow==2.13.0` or `tensorflow==2.14.0`.

```cmd
pip install tf-slim
pip install tf-keras
pip install streamlit
pip install opencv-python
pip install pillow
pip install numpy
pip install scipy
pip install scikit-image
pip install plotly
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install chumpy==0.70
```

Wait for all packages to finish installing before moving on.

### Step 3.6 — Verify the installation

```cmd
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
python -c "import streamlit; print('Streamlit OK')"
python -c "import cv2; print('OpenCV OK')"
```

All three should print without errors.

---

## 4. Apply Required Compatibility Patches

> **This section is mandatory.** The AI model was built with TensorFlow 1.x but the installed version is TF 2.x. These patches make them compatible. You only need to do this once.

### Patch 4.1 — Fix `chumpy` (3D body model library)

`chumpy` is a library used internally by the SMPL body model. It was written for old Python and needs two fixes.

**Fix A — Remove broken imports**

Open this file in Notepad (adjust the path to your Python version if needed):
```
C:\tbm\Lib\site-packages\chumpy\__init__.py
```

Find this line (near the top):
```python
from numpy import bool, int, float, complex, object, unicode, str, nan, inf
```

**Replace it with:**
```python
from numpy import nan, inf
```

Save and close.

**Fix B — Fix removed Python function**

Open:
```
C:\tbm\Lib\site-packages\chumpy\ch.py
```

Use Notepad's Find & Replace (Ctrl+H):
- Find: `inspect.getargspec`
- Replace: `inspect.getfullargspec`

Click "Replace All". It should replace 2 occurrences. Save and close.

---

### Patch 4.2 — Create TF1 compatibility layer for `tf_keras`

This patch makes old TensorFlow 1.x style neural network layers work correctly in TF 2.x. It ensures the AI model can load its pre-trained weights from checkpoint files.

**Step A — Create the folder**

Open Command Prompt (with `C:\tbm` active) and run:
```cmd
mkdir C:\tbm\Lib\site-packages\tf_keras\legacy_tf_layers
```

**Step B — Create the `__init__.py` file**

Create a new file at:
```
C:\tbm\Lib\site-packages\tf_keras\legacy_tf_layers\__init__.py
```

Paste this exact content:
```python
from . import normalization
from .normalization import BatchNormalization, batch_normalization
```

Save and close.

**Step C — Create the `normalization.py` file**

Create a new file at:
```
C:\tbm\Lib\site-packages\tf_keras\legacy_tf_layers\normalization.py
```

Paste this entire content exactly as shown:

```python
import tensorflow.compat.v1 as tf

_TF1_ONLY_KWARGS = frozenset({
    '_reuse', '_scope', 'fused', 'adjustment',
    'renorm', 'renorm_clipping', 'renorm_momentum',
    'zero_debias_moving_mean', 'param_regularizers',
    'updates_collections', 'variables_collections', 'outputs_collections',
    'activation_fn', 'is_training', 'data_format', 'name',
})

_VALID_INIT_KWARGS = frozenset({
    'axis', 'momentum', 'epsilon', 'center', 'scale',
    'beta_initializer', 'gamma_initializer',
    'moving_mean_initializer', 'moving_variance_initializer',
    'beta_regularizer', 'gamma_regularizer',
    'trainable', 'dtype',
})


class BatchNormalization:
    def __init__(self, axis=-1, momentum=0.99, epsilon=0.001,
                 center=True, scale=True, **kwargs):
        for k in list(kwargs.keys()):
            if k not in _VALID_INIT_KWARGS:
                kwargs.pop(k)

        self._axis = axis
        self._momentum = float(momentum)
        self._epsilon = float(epsilon)
        self._center = center
        self._scale = scale
        self._trainable = kwargs.get('trainable', True)
        self._dtype = kwargs.get('dtype', tf.float32)

        self._beta_init = kwargs.get('beta_initializer', tf.zeros_initializer())
        self._gamma_init = kwargs.get('gamma_initializer', tf.ones_initializer())
        self._mean_init = kwargs.get('moving_mean_initializer', tf.zeros_initializer())
        self._var_init = kwargs.get('moving_variance_initializer', tf.ones_initializer())

    def __call__(self, inputs, training=False, **kwargs):
        shape = inputs.get_shape()
        axis = self._axis
        ndim = len(shape)
        if axis < 0:
            axis = ndim + axis
        dim = shape[axis].value

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self._scale:
                self.gamma = tf.get_variable(
                    'gamma', shape=[dim], dtype=self._dtype,
                    initializer=self._gamma_init,
                    trainable=self._trainable)
            else:
                self.gamma = None

            if self._center:
                self.beta = tf.get_variable(
                    'beta', shape=[dim], dtype=self._dtype,
                    initializer=self._beta_init,
                    trainable=self._trainable)
            else:
                self.beta = None

            self.moving_mean = tf.get_variable(
                'moving_mean', shape=[dim], dtype=self._dtype,
                initializer=self._mean_init, trainable=False)

            self.moving_variance = tf.get_variable(
                'moving_variance', shape=[dim], dtype=self._dtype,
                initializer=self._var_init, trainable=False)

        outputs = tf.nn.batch_normalization(
            inputs,
            mean=self.moving_mean,
            variance=self.moving_variance,
            offset=self.beta,
            scale=self.gamma,
            variance_epsilon=self._epsilon,
        )
        return outputs

    def apply(self, inputs, training=False, **kwargs):
        return self(inputs, training=training)


def batch_normalization(inputs, **kwargs):
    is_training = kwargs.get('is_training', False)
    valid = {k: v for k, v in kwargs.items() if k in _VALID_INIT_KWARGS}
    layer = BatchNormalization(**valid)
    return layer(inputs, training=is_training)
```

Save and close.

---

## 5. Running the Web Interface

### Step 5.1 — Navigate to the project folder

Open Command Prompt and activate the environment:
```cmd
C:\tbm\Scripts\activate
cd C:\body-ai
```

Replace `C:\body-ai` with wherever you extracted the project files.

### Step 5.2 — Start the app

```cmd
streamlit run demo_app.py
```

After about 10–20 seconds you will see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should open automatically. If it does not, open your browser and go to:
```
http://localhost:8501
```

> **First run with Photo or Video input takes 30–60 seconds** because the AI model loads into memory. Subsequent runs are faster.

### Step 5.3 — Stop the app

Press `Ctrl + C` in the Command Prompt window.

---

## 6. How to Use the App

### Sidebar — Input Mode

In the left sidebar, choose one of four input modes:

#### Mode 1: Sample Subject (for testing)
- Requires having sample data in a `sample_data/` folder
- Select a subject from the dropdown, click **Extract Measurements**
- Shows measurements + 3D model + ground truth comparison

#### Mode 2: Upload `_smpld.json` (highest accuracy — 96.3%)
- For use with professional 3D body scanners
- Upload a `_smpld.json` file exported from the scanner
- Click **Extract Measurements**
- Results appear instantly (no AI inference needed — data is in the file)

#### Mode 3: Upload Photo (81.5% accuracy)
- Upload any JPG or PNG photo
- **Best results:** person standing straight, full body visible, plain background
- Enter the person's real height in cm (this is required for scale)
- Click **Extract Measurements**
- Takes 30–60 seconds on first run

#### Mode 4: Upload Video (81.5% accuracy)
- Upload MP4, AVI, MOV, MKV, or WebM video
- Use the slider to select the best frame (person standing straight, arms slightly away from body)
- Enter the person's real height in cm
- Click **Extract Measurements**

### Sidebar — Other Settings

| Toggle | What it does |
|---|---|
| **Guide Mode** | Shows step-by-step explanations in the interface |
| **Apply Calibration** | Improves accuracy using trained correction factors (recommended ON) |
| **Show Ground Truth** | Compare AI results vs tape-measure values (only for sample subjects) |
| **Show 3D Model** | Display interactive 3D body viewer |

### Results Tabs

After measurements are extracted, four tabs appear:

| Tab | Contents |
|---|---|
| **Measurements** | All 31 measurements grouped by category (chest, waist, etc.) |
| **Charts** | Bar chart visual comparison |
| **3D Model** | Interactive 3D body you can rotate and zoom |
| **Export** | Download results as JSON or CSV file |

---

## 7. Running the REST API Server (Optional)

If you want to connect to this system from your own software (website, mobile app, etc.), you can run it as an API.

### Start the API server

```cmd
C:\tbm\Scripts\activate
cd C:\body-ai
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API is now running at: `http://localhost:8000`

### API Endpoints

#### Health Check
```
GET http://localhost:8000/health
```
Returns `{"status": "ok"}` if the server is running.

#### Measure from 3D Scan file
```
POST http://localhost:8000/measure/scan
```
Form field: `file` (upload the `_smpld.json` file)

#### Measure from Photo
```
POST http://localhost:8000/measure/photo
```
Form fields:
- `file` — JPG or PNG image file
- `height_cm` — Person's height in cm (e.g., `170`)

#### Measure from Video
```
POST http://localhost:8000/measure/video
```
Form fields:
- `file` — video file (MP4, AVI, MOV, etc.)
- `height_cm` — Person's height in cm
- `frame_index` — Which frame to use (e.g., `0` for first frame)

### Example using curl

```bash
# Health check
curl http://localhost:8000/health

# Measure from photo
curl -X POST http://localhost:8000/measure/photo \
  -F "file=@/path/to/photo.jpg" \
  -F "height_cm=170"

# Measure from 3D scan
curl -X POST http://localhost:8000/measure/scan \
  -F "file=@/path/to/scan_smpld.json"
```

### Example API Response

```json
{
  "method": "photo_ai",
  "accuracy": "81.5% (estimated)",
  "measurement_count": 11,
  "measurements": {
    "height": {
      "value_cm": 170.0,
      "chinese": "模型身高"
    },
    "chest circumference": {
      "value_cm": 94.3,
      "chinese": "胸围"
    },
    "waist circumference": {
      "value_cm": 78.1,
      "chinese": "中腰围"
    }
  }
}
```

### Interactive API Documentation

Open this in your browser while the server is running:
```
http://localhost:8000/docs
```
This shows a full interactive form where you can test all endpoints.

---

## 8. Troubleshooting Common Errors

### Error: `No sample subjects found in sample_data/`

**Cause:** The `sample_data/` folder does not exist or is empty.

**Solution:** This is normal if you have no 3D scan data. Use **Upload Photo** or **Upload Video** mode instead. If you have scan data, place it in a `sample_data/` folder inside the project directory.

---

### Error: `ModuleNotFoundError: No module named 'tf_slim'`

**Solution:**
```cmd
C:\tbm\Scripts\activate
pip install tf-slim
```

---

### Error: `ModuleNotFoundError: No module named 'chumpy'`

**Solution:**
```cmd
C:\tbm\Scripts\activate
pip install chumpy==0.70
```
Then apply Patch 4.1 from this guide.

---

### Error: `ModuleNotFoundError: No module named 'tf_keras'`

**Solution:**
```cmd
C:\tbm\Scripts\activate
pip install tf-keras
```

---

### Error: `ModuleNotFoundError: No module named 'tf_keras.legacy_tf_layers'`

**Solution:** Apply Patch 4.2 from Section 4 of this guide. You need to create both files manually.

---

### Error: `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'`

**Cause:** Pillow version 10+ removed the old `ANTIALIAS` name.

**Solution:**
```cmd
pip install "pillow>=10.0"
```
This version is already correct — the project code has been updated to use the new `LANCZOS` name. If you see this error, make sure you have the latest version of the project files.

---

### Error: `AttributeError: module 'numpy' has no attribute 'int'` (or `np.float`, `np.bool`)

**Cause:** NumPy 1.24+ removed old type aliases.

**Solution:** Make sure you are using the latest version of the project files from GitHub. All these have been fixed.

---

### App is very slow on first Photo/Video run

**Explanation:** The AI loads two large neural networks (DeepLab and HMR) into memory on the first run. This takes 30–90 seconds. All subsequent runs in the same session are fast (2–5 seconds).

**If you keep restarting the app**, consider keeping it running and just uploading different files.

---

### Error: TensorFlow not found or wrong version

Make sure you always activate the virtual environment first:
```cmd
C:\tbm\Scripts\activate
```

Check TF is installed:
```cmd
python -c "import tensorflow as tf; print(tf.__version__)"
```

If not found, install it:
```cmd
pip install tensorflow==2.15.0
```

---

### Port 8501 already in use

Another app is using port 8501. Run Streamlit on a different port:
```cmd
streamlit run demo_app.py --server.port 8502
```
Then open `http://localhost:8502` in your browser.

---

### Browser does not open automatically

Manually open your browser and go to:
```
http://localhost:8501
```

---

## 9. File Structure Reference

```
C:\body-ai\                         ← Project root (your project folder)
│
├── demo_app.py                     ← Main web interface (run this with streamlit)
├── measurement_api.py              ← Python API (used internally by the app)
├── api_server.py                   ← REST API server (run with uvicorn)
├── inference.py                    ← DeepLab background removal
├── demo.py                         ← HMR (Human Mesh Recovery) AI model
├── extract_measurements.py         ← Convert 3D vertices → measurements in cm
├── utils.py                        ← Measurement name constants
├── networks.py                     ← Neural network architecture definitions
│
├── data/                           ← Body measurement control point data
│   └── customBodyPoints.txt        ← Required: defines where to measure
│
├── models/                         ← Pre-trained AI model weights
│   ├── model.ckpt-*                ← HMR model checkpoint (required for photo/video)
│   └── deeplab/                    ← DeepLab segmentation model
│
├── src/                            ← Core AI source code
│   ├── tf_smpl/                    ← SMPL 3D body model (TensorFlow)
│   └── util/                       ← Image processing utilities
│
├── sample_data/                    ← (Optional) pre-loaded test subjects
│   └── subject_name/
│       └── model/
│           └── subject_smpld.json  ← 3D scan file
│
├── SETUP_GUIDE.md                  ← This file
└── DEPLOYMENT.md                   ← Server deployment documentation
```

### Key files that must exist

| File | Required for |
|---|---|
| `data/customBodyPoints.txt` | All measurement modes |
| `models/model.ckpt-*` | Photo and Video modes |
| `models/deeplab/` | Photo and Video modes |
| `src/tf_smpl/neutral_smpl_with_cocoplus_reg.pkl` | Photo and Video modes |

---

## Quick Start Checklist

Use this checklist each time you set up on a new machine:

- [ ] Python 3.11 installed with "Add to PATH" checked
- [ ] Created virtual environment at `C:\tbm` using `python -m venv C:\tbm`
- [ ] Activated environment: `C:\tbm\Scripts\activate`
- [ ] Installed all packages (Section 3.5)
- [ ] Applied Patch 4.1 — fixed `chumpy/__init__.py` and `chumpy/ch.py`
- [ ] Applied Patch 4.2 — created `legacy_tf_layers/__init__.py` and `normalization.py`
- [ ] Ran `streamlit run demo_app.py` from the project folder
- [ ] Browser opened at `http://localhost:8501`

---

## Getting Help

If you encounter an error not listed here:

1. Copy the **full error message** from the terminal
2. Note which **step** you were on
3. Note your **Python version** (`python --version`) and **OS version**
4. Send this information to your contact for support

---

*Document version: June 2026*
*Project: Human Body Measurements using Computer Vision*
