"""
Body Measurement AI — Live Demo
================================
Streamlit web interface for demonstrating AI-powered body measurement extraction.

Run:  streamlit run demo_app.py
"""

import streamlit as st
import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
import tempfile
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measurement_api import MeasurementAPI

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Body Measurement AI",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GUIDE / WALKTHROUGH CONTENT
# ============================================================
GUIDE_STEPS = {
    "welcome": """
### Welcome to Body Measurement AI

This system **automatically extracts 31 body measurements** from a 3D body scan with **96.3% accuracy**.

**How it works (3 simple steps):**
1. **Select a person** — Pick a subject from the dropdown (or upload a 3D scan file)
2. **Click "Extract Measurements"** — The AI analyzes the 3D body model
3. **View results** — See all measurements in cm, charts, 3D model, and export data

**What is this for?**
- Clothing & tailoring — get precise body measurements without a tape measure
- Fitness tracking — monitor body changes over time
- Fashion e-commerce — recommend sizes based on real measurements
""",
    "sidebar": """
**Sidebar Settings Explained:**
- **Sample Subject** = pre-loaded test subjects with known measurements
- **Upload _smpld.json** = bring your own 3D scan file (96.3% accuracy)
- **Upload Photo** = upload a JPG/PNG photo + enter your height (81.5% accuracy)
- **Front + Side Photos** = upload front AND side photos for ~93% accuracy
- **Upload Video** = upload a video + pick a frame + enter your height (81.5% accuracy)
- **Apply Calibration** = improves accuracy using AI correction factors
- **Show Ground Truth** = compare AI predictions vs real tape-measure values
- **Show 3D Model** = display the 3D body you can rotate and zoom
""",
    "subject": """**Step 1:** Select a person from the dropdown below. Each subject has a 3D body scan from a professional body scanner.""",
    "extract": """**Step 2:** Click the red button to run the AI measurement extraction. It takes about 1 second.""",
    "results_summary": """**Step 3 — Results Summary:** These 4 numbers show: how many measurements were extracted, overall accuracy, the person's height, and which method was used (3D Scan vs Photo).""",
    "measurements_tab": """
**Measurements Tab:** All 31 body measurements grouped by category:
- **Core Dimensions** = height, shoulder width, sleeve length
- **Circumferences** = neck, chest, waist, hip, thigh, calf, etc.
- **Lengths & Widths** = torso lengths, pant measurements, crotch

Each card shows:
- **Chinese + English name** of the measurement
- **AI prediction in cm** (bold blue number)
- **GT (Ground Truth)** = the real tape-measure value for comparison, with error in brackets
- Green = very accurate (< 1.5 cm error), Orange = good (< 3 cm), Red = needs improvement
""",
    "charts_tab": """
**Charts Tab:** Visual comparison of AI predictions vs real measurements.
- Blue bars = AI predictions
- Green bars = actual tape-measure values (Ground Truth)
- The closer they match, the more accurate the AI is
- Statistics at the bottom show mean error and % within 2cm tolerance
""",
    "3d_tab": """
**3D Model Tab:** Interactive 3D body visualization.
- **Drag** to rotate the body
- **Scroll** to zoom in/out
- **Right-click drag** to pan
- The mesh has 35,490 vertices representing the body surface
""",
    "export_tab": """
**Export Tab:** Download the measurement results:
- **JSON** = structured data for software integration
- **CSV** = spreadsheet-compatible format for Excel/Google Sheets
""",
}

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1a73e8, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        border-left: 4px solid #1a73e8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .accuracy-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .accuracy-high { background: #e8f5e9; color: #2e7d32; }
    .accuracy-med { background: #fff3e0; color: #e65100; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 0.3rem;
        margin: 1rem 0 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .guide-box {
        background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
        border: 1px solid #90caf9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1f2d3d;
    }
    .guide-box, .guide-box p, .guide-box li,
    .guide-box ul, .guide-box ol, .guide-box span {
        color: #1f2d3d !important;
    }
    .guide-box strong, .guide-box b {
        color: #0d2538 !important;
    }
    .guide-box h3 {
        color: #1565c0 !important;
        margin-top: 0;
    }
    .guide-step {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #3d2c00;
    }
    .guide-step, .guide-step p, .guide-step li, .guide-step span {
        color: #3d2c00 !important;
    }
    .guide-step strong, .guide-step b {
        color: #1f1500 !important;
    }
</style>
""", unsafe_allow_html=True)


def guide_box(text):
    """Render a guide explanation box."""
    st.markdown(f'<div class="guide-box">{text}</div>', unsafe_allow_html=True)


def guide_step(text):
    """Render a guide step hint."""
    st.markdown(f'<div class="guide-step">{text}</div>', unsafe_allow_html=True)


# ============================================================
# MEASUREMENT CATEGORIES for organized display
# ============================================================
CATEGORIES = {
    "📏 Core Dimensions": [
        "height", "back shoulder breadth", "front shoulder breadth",
        "sleeve length", "left sleeve length",
    ],
    "🔵 Circumferences — Upper Body": [
        "neck circumference", "chest circumference",
        "bicep left circumference", "bicep right circumference",
        "wrist left circumference",
    ],
    "🟢 Circumferences — Torso": [
        "waist circumference", "mid waist circumference",
        "pants waist circumference", "hip circumference",
    ],
    "🟡 Circumferences — Lower Body": [
        "thigh left circumference", "thigh right circumference",
        "knee circumference", "calf left circumference",
        "calf right circumference",
    ],
    "📐 Lengths & Widths": [
        "neck width", "front chest breadth", "back chest breadth",
        "front waist length", "back waist length", "back mid waist length",
        "back clothing length", "back waist height",
        "pant height", "leg height", "O to under hip", "open crotch",
    ],
}


@st.cache_resource
def get_api():
    """Initialize and cache the measurement API."""
    return MeasurementAPI()


def discover_sample_subjects():
    """Find all sample subjects with _smpld.json files."""
    subjects = {}
    if not os.path.exists("sample_data"):
        return subjects
    data_dirs = [
        os.path.join("sample_data", d) for d in os.listdir("sample_data")
        if os.path.isdir(os.path.join("sample_data", d))
    ]

    for data_dir in data_dirs:
        try:
            for subj_name in sorted(os.listdir(data_dir)):
                subj_path = os.path.join(data_dir, subj_name)
                if not os.path.isdir(subj_path):
                    continue
                # Find _smpld.json
                for root, dirs, files in os.walk(subj_path):
                    for f in files:
                        if f.endswith('_smpld.json'):
                            subjects[subj_name] = os.path.join(root, f)
                            break
        except (PermissionError, OSError):
            continue

    return subjects


def find_obj_file(smpld_path):
    """Find the .obj file in the same directory as the smpld.json."""
    obj_dir = os.path.dirname(smpld_path)
    for f in os.listdir(obj_dir):
        if f.endswith('.obj'):
            return os.path.join(obj_dir, f)
    return None


def find_ground_truth(smpld_path):
    """Find the ground truth .txt file for a subject."""
    import re
    # Go up two levels from model dir to subject dir
    subj_dir = os.path.dirname(os.path.dirname(smpld_path))
    for f in os.listdir(subj_dir):
        if f.endswith('.txt') and not f.startswith('.'):
            gt_path = os.path.join(subj_dir, f)
            if os.path.isfile(gt_path):
                measurements = {}
                with open(gt_path, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        line = line.strip()
                        parts = re.split(r'[:：]', line, maxsplit=1)
                        if len(parts) == 2:
                            try:
                                measurements[parts[0].strip()] = float(parts[1].strip())
                            except ValueError:
                                pass
                return measurements
    return None


def create_3d_body_visualization(landmarks):
    """Create a 3D scatter plot of body landmarks using Plotly."""
    if not landmarks:
        return None

    xs, ys, zs, names = [], [], [], []
    for name, coords in landmarks.items():
        if isinstance(coords, list) and len(coords) == 3:
            xs.append(coords[0])
            ys.append(coords[2])  # swap y/z for upright view
            zs.append(coords[1])
            names.append(name)

    if not xs:
        return None

    fig = go.Figure(data=[go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=3, color=zs, colorscale='Plasma', opacity=0.8),
        text=names,
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>',
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=0.5, z=0.5)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
    )

    return fig


def create_3d_obj_visualization(obj_path):
    """Create a 3D mesh from an OBJ file using Plotly."""
    vertices = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f' and len(parts) >= 4:
                # Handle face indices (may have v/vt/vn format)
                face_verts = []
                for p in parts[1:4]:
                    face_verts.append(int(p.split('/')[0]) - 1)
                faces.append(face_verts)

    if not vertices:
        return None

    verts = np.array(vertices)
    face_arr = np.array(faces)

    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 2],
        z=verts[:, 1],
        i=face_arr[:, 0],
        j=face_arr[:, 1],
        k=face_arr[:, 2],
        color='#ffb6c1',
        opacity=0.7,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo='skip',
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=0.3, z=0.3)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
    )

    return fig


def create_measurement_chart(measurements):
    """Create a horizontal bar chart of measurements."""
    # Filter to circumferences for a nice visual
    circ_keys = [k for k in measurements if 'circumference' in k or 'breadth' in k]
    if not circ_keys:
        circ_keys = list(measurements.keys())[:12]

    names = []
    values = []
    for k in circ_keys:
        m = measurements[k]
        names.append(m.get('chinese', k))
        values.append(m['value_cm'])

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color='#1a73e8',
        text=[f'{v} cm' for v in values],
        textposition='outside',
    ))

    fig.update_layout(
        xaxis_title='Centimeters (cm)',
        yaxis=dict(autorange='reversed'),
        margin=dict(l=0, r=50, t=10, b=40),
        height=max(300, len(names) * 35),
    )

    return fig


def render_measurements_table(measurements, gt=None):
    """Render measurements as categorized cards."""
    cn_to_en = {}
    for en_key, info in measurements.items():
        cn_to_en[info.get('chinese', '')] = en_key

    # Ground truth mapping
    GT_MAP = {
        "模型身高": "height", "领围": "neck circumference", "颈宽": "neck width",
        "胸围": "chest circumference", "中腰围": "waist circumference",
        "肚围": "pants waist circumference", "肩宽": "back shoulder breadth",
        "前肩宽": "front shoulder breadth", "前胸宽": "front chest breadth",
        "后背宽": "back chest breadth", "左袖长": "left sleeve length",
        "袖丈": "sleeve length", "左上臂围": "bicep left circumference",
        "右上臂围": "bicep right circumference", "手腕围": "wrist left circumference",
        "前腰节长": "front waist length", "后中腰节长": "back waist length",
        "后腰节长": "back mid waist length", "后衣长(到臀下)": "back clothing length",
        "裤腰围": "mid waist circumference", "下臀围": "hip circumference",
        "左大腿围": "thigh left circumference", "右大腿围": "thigh right circumference",
        "膝围": "knee circumference", "左小腿围": "calf left circumference",
        "右小腿围": "calf right circumference", "N点～臀下": "leg height",
        "O点～臀下": "O to under hip", "左裤长": "pant height",
        "通裆": "open crotch", "后腰高": "back waist height",
    }

    for cat_name, cat_keys in CATEGORIES.items():
        displayed = [(k, measurements[k]) for k in cat_keys if k in measurements]
        if not displayed:
            continue

        st.markdown(f'<div class="section-header">{cat_name}</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(displayed), 4))

        for i, (en_key, m) in enumerate(displayed):
            col = cols[i % len(cols)]
            with col:
                gt_val = None
                gt_html = ""
                if gt:
                    # Find gt value by reverse mapping
                    for cn_gt, en_gt in GT_MAP.items():
                        if en_gt == en_key and cn_gt in gt:
                            gt_val = gt[cn_gt]
                            err = abs(m['value_cm'] - gt_val)
                            color = "#2e7d32" if err <= 1.5 else ("#e65100" if err <= 3 else "#c62828")
                            gt_html = f'<div style="font-size:0.75rem;color:{color}">GT: {gt_val} cm (±{err:.1f})</div>'
                            break

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{m.get('chinese', '')} · {en_key}</div>
                    <div class="metric-value">{m['value_cm']} cm</div>
                    {gt_html}
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-header">📐 Body Measurement AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;margin-top:-0.5rem">AI-powered body measurement extraction from 3D scans — 96.3% accuracy</p>', unsafe_allow_html=True)

    api = get_api()

    # Guide mode state
    if 'guide_mode' not in st.session_state:
        st.session_state['guide_mode'] = True

    # ---- SIDEBAR ----
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/body-measurements.png", width=60)
        st.title("Settings")

        guide_mode = st.toggle("🎓 Guide Mode", value=st.session_state['guide_mode'],
                               help="Show step-by-step explanations throughout the interface")
        st.session_state['guide_mode'] = guide_mode

        if guide_mode:
            st.info("💡 Guide Mode is ON — explanations appear throughout the interface. Turn off for a clean view.")

        st.divider()

        mode = st.radio("Input Mode", [
            "📁 Sample Subject",
            "📤 Upload _smpld.json",
            "📷 Upload Photo",
            "📸 Front + Side Photos (~93%)",
            "🎬 Upload Video",
        ], index=0)

        apply_cal = st.toggle("Apply Calibration", value=True,
                              help="Apply trained calibration factors (recommended)")

        show_gt = st.toggle("Show Ground Truth", value=True,
                            help="Compare predictions against manual measurements")

        show_3d = st.toggle("Show 3D Model", value=True,
                            help="Display interactive 3D body visualization")

        if guide_mode:
            st.divider()
            st.markdown(GUIDE_STEPS['sidebar'])

        st.divider()
        st.markdown("### 📊 System Info")
        st.markdown(f"""
        - **Model**: SMPL+D (35,490 vertices)
        - **Accuracy**: 96.3% (calibrated)
        - **Measurements**: 31 types
        - **Calibration**: LOO-CV on 36 subjects
        """)

    # ---- MAIN CONTENT ----
    # ---- HOW IT WORKS (Guide Mode) ----
    if guide_mode:
        with st.expander("📖 How It Works — Click to expand/collapse", expanded=True):
            st.markdown(GUIDE_STEPS['welcome'])

    result = None
    smpld_path = None
    subject_name = "Unknown"

    if mode == "📁 Sample Subject":
        subjects = discover_sample_subjects()
        if not subjects:
            st.warning("No sample subjects found in sample_data/")
            return

        if guide_mode:
            guide_step(GUIDE_STEPS['subject'])

        selected = st.selectbox(
            "Select a subject",
            list(subjects.keys()),
            format_func=lambda x: f"👤 {x}",
        )

        if selected:
            smpld_path = subjects[selected]
            subject_name = selected

            if guide_mode:
                guide_step(GUIDE_STEPS['extract'])

            if st.button("🔬 Extract Measurements", type="primary", use_container_width=True):
                with st.spinner("Analyzing 3D body model..."):
                    result = api.measure_from_smpld(smpld_path, apply_calibration=apply_cal)
                st.session_state['result'] = result
                st.session_state['smpld_path'] = smpld_path
                st.session_state['subject_name'] = subject_name

            if 'result' in st.session_state and st.session_state.get('subject_name') == subject_name:
                result = st.session_state['result']
                smpld_path = st.session_state['smpld_path']

    elif mode == "📤 Upload _smpld.json":
        uploaded = st.file_uploader("Upload a _smpld.json file", type=['json'])
        if uploaded:
            subject_name = uploaded.name.replace('_smpld.json', '')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp:
                content = uploaded.read().decode('utf-8')
                tmp.write(content)
                tmp_path = tmp.name

            if st.button("🔬 Extract Measurements", type="primary", use_container_width=True):
                with st.spinner("Analyzing 3D body model..."):
                    result = api.measure_from_smpld(tmp_path, apply_calibration=apply_cal)
                st.session_state['result'] = result
                st.session_state['smpld_path'] = None
                st.session_state['subject_name'] = subject_name

            if 'result' in st.session_state:
                result = st.session_state['result']

    elif mode == "📷 Upload Photo":
        uploaded_photo = st.file_uploader(
            "Upload a photo of the person",
            type=['jpg', 'jpeg', 'png'],
            help="Best results: full body visible, person standing upright, plain background",
        )
        height_input = st.number_input(
            "Your height (cm)", min_value=100, max_value=250, value=170, step=1
        )

        if uploaded_photo:
            # Show preview
            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(uploaded_photo, caption=uploaded_photo.name, use_container_width=True)
            with col_info:
                st.markdown(f"**File:** {uploaded_photo.name}")
                st.markdown(f"**Height input:** {height_input} cm")
                st.markdown("**Tips for best results:**")
                st.markdown("- Full body in frame\n- Person standing straight\n- Plain / light background")

            subject_name = os.path.splitext(uploaded_photo.name)[0]

            if guide_mode:
                guide_step(GUIDE_STEPS['extract'])

            if st.button("🔬 Extract Measurements", type="primary", use_container_width=True):
                suffix = os.path.splitext(uploaded_photo.name)[1].lower() or '.jpg'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_photo.read())
                    tmp_path = tmp.name
                with st.spinner("Running AI inference — this may take 30–60 s on first run…"):
                    result = api.measure_from_image(tmp_path, height_cm=float(height_input))
                os.unlink(tmp_path)
                st.session_state['result'] = result
                st.session_state['smpld_path'] = None
                st.session_state['subject_name'] = subject_name

            if 'result' in st.session_state and st.session_state.get('subject_name') == subject_name:
                result = st.session_state['result']

    elif mode == "📸 Front + Side Photos (~93%)":
        st.info("Upload **two photos**: one facing the camera directly (front view) and one standing 90° sideways (side view). This resolves depth ambiguity and raises accuracy to ~93%.")

        col_f, col_s = st.columns(2)
        with col_f:
            st.markdown("**Front Photo** — person faces the camera directly")
            uploaded_front = st.file_uploader(
                "Upload front photo",
                type=['jpg', 'jpeg', 'png'],
                key="front_photo",
            )
            if uploaded_front:
                st.image(uploaded_front, caption="Front view", use_container_width=True)

        with col_s:
            st.markdown("**Side Photo** — person stands sideways (90°)")
            uploaded_side = st.file_uploader(
                "Upload side photo",
                type=['jpg', 'jpeg', 'png'],
                key="side_photo",
            )
            if uploaded_side:
                st.image(uploaded_side, caption="Side view", use_container_width=True)

        height_input = st.number_input(
            "Your height (cm)", min_value=100, max_value=250, value=170, step=1,
            key="dual_height",
        )

        st.markdown("**Photo tips for best accuracy:**")
        st.markdown(
            "- Full body in frame (head to feet)\n"
            "- Person standing straight, arms slightly away from body\n"
            "- Plain / light background\n"
            "- Good lighting, no heavy shadows\n"
            "- Same outfit in both photos"
        )

        if uploaded_front and uploaded_side:
            subject_name = os.path.splitext(uploaded_front.name)[0]

            if guide_mode:
                guide_step(GUIDE_STEPS['extract'])

            if st.button("🔬 Extract Measurements (Front + Side)", type="primary", use_container_width=True):
                front_suffix = os.path.splitext(uploaded_front.name)[1].lower() or '.jpg'
                side_suffix  = os.path.splitext(uploaded_side.name)[1].lower() or '.jpg'

                with tempfile.NamedTemporaryFile(delete=False, suffix=front_suffix) as tf:
                    tf.write(uploaded_front.read())
                    front_tmp = tf.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=side_suffix) as ts:
                    ts.write(uploaded_side.read())
                    side_tmp = ts.name

                try:
                    with st.spinner("Running AI inference on both photos — this may take 60–90 s on first run…"):
                        result = api.measure_from_two_images(
                            front_tmp, side_tmp, height_cm=float(height_input)
                        )
                finally:
                    for p in (front_tmp, side_tmp):
                        if os.path.exists(p):
                            os.unlink(p)

                st.session_state['result'] = result
                st.session_state['smpld_path'] = None
                st.session_state['subject_name'] = subject_name

            if 'result' in st.session_state and st.session_state.get('subject_name') == subject_name:
                result = st.session_state['result']

        elif uploaded_front or uploaded_side:
            st.warning("Please upload **both** a front photo and a side photo to continue.")

    elif mode == "🎬 Upload Video":
        uploaded_video = st.file_uploader(
            "Upload a video file", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v']
        )
        height_input = st.number_input(
            "Your height (cm)", min_value=100, max_value=250, value=170, step=1
        )

        if uploaded_video:
            # Persist video path across Streamlit reruns using session state
            vid_key = f"vid_{uploaded_video.name}_{uploaded_video.size}"
            if st.session_state.get('video_key') != vid_key:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_video.name)[1]
                ) as tmp:
                    tmp.write(uploaded_video.read())
                    st.session_state['video_tmp_path'] = tmp.name
                st.session_state['video_key'] = vid_key

            tmp_video_path = st.session_state['video_tmp_path']

            # Read video metadata
            cap = cv2.VideoCapture(tmp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            duration = total_frames / fps if fps > 0 else 0
            st.info(
                f"Video loaded — **{total_frames}** frames · **{fps:.1f}** fps · **{duration:.1f}s**"
            )

            analysis_mode = st.radio(
                "Analysis mode",
                ["🤖 Auto multi-angle (~93%)", "🎯 Single frame (~81.5%)"],
                index=0,
                help="Auto multi-angle samples many frames, finds the best front-facing "
                     "and side-facing poses automatically, and blends them. For best "
                     "results, have the person slowly turn during the video.",
            )

            subject_name = uploaded_video.name

            if analysis_mode.startswith("🤖"):
                # ---- AUTO MULTI-ANGLE MODE ----
                st.markdown(
                    "**How to record:** have the person stand ~2–3 m from the camera "
                    "and **slowly rotate 90–180°** so the video captures both a front "
                    "view and a side view. The AI picks the best of each automatically."
                )
                num_samples = st.slider(
                    "Frames to analyse (more = slower but more thorough)",
                    min_value=4, max_value=24, value=12, step=2,
                )

                if guide_mode:
                    guide_step(GUIDE_STEPS['extract'])

                if st.button("🔬 Extract Measurements (Auto Multi-Angle)",
                             type="primary", use_container_width=True):
                    prog = st.progress(0.0, text="Starting multi-angle analysis…")

                    def _cb(frac, msg):
                        prog.progress(min(frac, 1.0), text=msg)

                    with st.spinner("Sampling and analysing multiple angles…"):
                        result = api.measure_from_video_multiangle(
                            tmp_video_path,
                            height_cm=float(height_input),
                            num_samples=num_samples,
                            progress_cb=_cb,
                        )
                    prog.empty()

                    st.success(
                        f"Analysed {result['frames_analyzed']} frames. "
                        f"Best front pose: frame {result['front_frame']} "
                        f"(score {result['front_score']}) · "
                        f"Best side pose: frame {result['side_frame']} "
                        f"(score {result['side_score']})."
                    )

                    st.session_state['result'] = result
                    st.session_state['smpld_path'] = None
                    st.session_state['subject_name'] = subject_name

                if 'result' in st.session_state and st.session_state.get('subject_name') == subject_name:
                    result = st.session_state['result']

                    # ---- Show the actual frames the AI selected ----
                    if result.get('front_frame') is not None and 'side_frame' in result:
                        st.markdown("#### 🖼️ Angles the AI selected from your video")

                        def _grab_frame(path, idx, max_w=420):
                            cap = cv2.VideoCapture(path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                            ok, fr = cap.read()
                            cap.release()
                            if not ok:
                                return None
                            h, w = fr.shape[:2]
                            if w > max_w:
                                scale = max_w / w
                                fr = cv2.resize(fr, (max_w, int(h * scale)))
                            return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

                        fps_v = result.get('fps', 0) or 0
                        fcol, scol = st.columns(2)
                        with fcol:
                            ff = _grab_frame(tmp_video_path, result['front_frame'])
                            ts = result['front_frame'] / fps_v if fps_v else 0
                            if ff is not None:
                                st.image(
                                    ff,
                                    caption=(f"FRONT view · frame {result['front_frame']} "
                                             f"· {ts:.1f}s · orientation {result['front_score']}"),
                                    use_container_width=True,
                                )
                        with scol:
                            sf = _grab_frame(tmp_video_path, result['side_frame'])
                            ts = result['side_frame'] / fps_v if fps_v else 0
                            if sf is not None:
                                st.image(
                                    sf,
                                    caption=(f"SIDE view · frame {result['side_frame']} "
                                             f"· {ts:.1f}s · orientation {result['side_score']}"),
                                    use_container_width=True,
                                )
                        st.caption(
                            f"Analysed {result['frames_analyzed']} frames sampled across "
                            f"{result['total_frames']} total · orientation score 1.0 = facing "
                            f"camera, 0.0 = side-on. Front + side blended for higher accuracy."
                        )

                        # Full gallery of every analysed angle
                        sampled = result.get('sampled_frames', [])
                        if sampled:
                            with st.expander(
                                f"🔍 See all {len(sampled)} analysed angles", expanded=False
                            ):
                                ncol = 4
                                rows = [sampled[i:i + ncol] for i in range(0, len(sampled), ncol)]
                                for row in rows:
                                    cols = st.columns(ncol)
                                    for c, item in zip(cols, row):
                                        with c:
                                            img = _grab_frame(tmp_video_path, item['index'])
                                            if img is not None:
                                                tag = ("FRONT" if item['index'] == result['front_frame']
                                                       else "SIDE" if item['index'] == result['side_frame']
                                                       else "")
                                                cap = (f"f{item['index']} · score {item['frontness']}"
                                                       + (f" · ✅{tag}" if tag else ""))
                                                st.image(img, caption=cap, use_container_width=True)

            else:
                # ---- SINGLE-FRAME MODE ----
                frame_idx = st.slider(
                    "Select frame to analyse",
                    min_value=0, max_value=max(0, total_frames - 1),
                    value=total_frames // 2,
                )

                # Live frame preview
                cap = cv2.VideoCapture(tmp_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    col_prev, col_info = st.columns([2, 1])
                    with col_prev:
                        st.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx} / {total_frames}",
                            use_container_width=True,
                        )
                    with col_info:
                        st.markdown(f"**Timestamp:** {frame_idx / fps:.2f}s")
                        st.markdown(f"**Height input:** {height_input} cm")
                        st.markdown(f"**Resolution:** {frame.shape[1]}×{frame.shape[0]}")

                if guide_mode:
                    guide_step(GUIDE_STEPS['extract'])

                if st.button("🔬 Extract Measurements", type="primary", use_container_width=True):
                    with st.spinner("Analysing selected frame..."):
                        result = api.measure_from_video(
                            tmp_video_path,
                            height_cm=float(height_input),
                            frame_index=frame_idx,
                        )
                    st.session_state['result'] = result
                    st.session_state['smpld_path'] = None
                    st.session_state['subject_name'] = subject_name

                if 'result' in st.session_state and st.session_state.get('subject_name') == subject_name:
                    result = st.session_state['result']

    # ---- DISPLAY RESULTS ----
    if result:
        measurements = result['measurements']

        if guide_mode:
            guide_step(GUIDE_STEPS['results_summary'])

        # Summary row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Measurements", result['measurement_count'])
        with col2:
            st.metric("Accuracy", result['accuracy'].split('(')[0].strip())
        with col3:
            height_m = measurements.get('height', {}).get('value_cm', 0)
            st.metric("Height", f"{height_m} cm")
        with col4:
            method_label = (
                "3D Scan"          if 'smpld'      in result['method'] else
                "Front+Side"       if 'dual'       in result['method'] else
                "Video Multi-Angle" if 'multiangle' in result['method'] else
                "Video AI"         if 'video'      in result['method'] else
                "Photo AI"
            )
            st.metric("Method", method_label)

        st.divider()

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Measurements", "📊 Charts", "🧍 3D Model", "📥 Export"])

        # ---- TAB 1: Measurements ----
        with tab1:
            if guide_mode:
                guide_box(GUIDE_STEPS['measurements_tab'])

            gt = None
            if show_gt and smpld_path:
                gt = find_ground_truth(smpld_path)

            render_measurements_table(measurements, gt)

        # ---- TAB 2: Charts ----
        with tab2:
            if guide_mode:
                guide_box(GUIDE_STEPS['charts_tab'])

            st.plotly_chart(create_measurement_chart(measurements), use_container_width=True)

            # Comparison chart if ground truth
            if show_gt and smpld_path:
                gt = find_ground_truth(smpld_path)
                if gt:
                    GT_MAP = {
                        "领围": "neck circumference", "胸围": "chest circumference",
                        "中腰围": "waist circumference", "下臀围": "hip circumference",
                        "肩宽": "back shoulder breadth", "手腕围": "wrist left circumference",
                        "左大腿围": "thigh left circumference", "膝围": "knee circumference",
                        "左袖长": "left sleeve length", "左小腿围": "calf left circumference",
                    }

                    pred_vals, gt_vals, labels = [], [], []
                    for cn_key, en_key in GT_MAP.items():
                        if cn_key in gt and en_key in measurements:
                            labels.append(cn_key)
                            pred_vals.append(measurements[en_key]['value_cm'])
                            gt_vals.append(gt[cn_key])

                    if labels:
                        st.markdown("### AI vs Ground Truth Comparison")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='AI Prediction', x=labels, y=pred_vals, marker_color='#1a73e8'))
                        fig.add_trace(go.Bar(name='Ground Truth', x=labels, y=gt_vals, marker_color='#2e7d32'))
                        fig.update_layout(barmode='group', yaxis_title='cm', height=400,
                                          margin=dict(t=30, b=60))
                        st.plotly_chart(fig, use_container_width=True)

                        errors = [abs(p - g) for p, g in zip(pred_vals, gt_vals)]
                        st.markdown(f"**Mean error: {np.mean(errors):.2f} cm** | "
                                    f"Max: {np.max(errors):.1f} cm | "
                                    f"Within 2cm: {sum(1 for e in errors if e <= 2)/len(errors)*100:.0f}%")

        # ---- TAB 3: 3D Model ----
        with tab3:
            if guide_mode:
                guide_box(GUIDE_STEPS['3d_tab'])

            if show_3d:
                obj_path = find_obj_file(smpld_path) if smpld_path else None

                if obj_path and os.path.exists(obj_path):
                    with st.spinner("Loading 3D mesh..."):
                        fig = create_3d_obj_visualization(obj_path)
                    if fig:
                        st.markdown("### Interactive 3D Body Model (35,490 vertices)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Drag to rotate · Scroll to zoom · Right-click to pan")
                    else:
                        st.info("Could not parse OBJ file.")
                elif result.get('landmarks'):
                    st.markdown("### 3D Body Landmarks (160 points)")
                    fig = create_3d_body_visualization(result['landmarks'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No 3D model available for this subject.")
            else:
                st.info("Enable '3D Model' in sidebar to view.")

        # ---- TAB 4: Export ----
        with tab4:
            if guide_mode:
                guide_box(GUIDE_STEPS['export_tab'])

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### JSON Export")
                export_data = {
                    'subject': subject_name,
                    'method': result['method'],
                    'accuracy': result['accuracy'],
                    'measurements': {
                        k: {'value_cm': v['value_cm'], 'name_cn': v['chinese']}
                        for k, v in measurements.items()
                    }
                }
                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    json_str,
                    file_name=f"{subject_name}_measurements.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.code(json_str[:600] + "\n  ...", language="json")

            with col_b:
                st.markdown("### CSV Export")
                csv_lines = ["Measurement,Chinese Name,Value (cm)"]
                for en_key, m in measurements.items():
                    csv_lines.append(f"{en_key},{m['chinese']},{m['value_cm']}")
                csv_str = "\n".join(csv_lines)
                st.download_button(
                    "⬇️ Download CSV",
                    csv_str,
                    file_name=f"{subject_name}_measurements.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.code(csv_str[:400] + "\n...", language="csv")


if __name__ == '__main__':
    main()
