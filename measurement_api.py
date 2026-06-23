"""
Unified Body Measurement API
============================
Provides body measurements via two pathways:

  Path A (Image → HMR): Lower accuracy (~81.5%), works from photos
  Path B (3D Model → SMPLD): High accuracy (~96.3%), uses client's pre-scanned 3D models

Usage:
    from measurement_api import MeasurementAPI
    api = MeasurementAPI()

    # Path B: From client's 3D model (recommended, highest accuracy)
    result = api.measure_from_smpld("path/to/_smpld.json")

    # Path A: From image (lower accuracy, no 3D scanner needed)
    result = api.measure_from_image("path/to/photo.jpg", height_cm=170)
"""


import json
import os
import sys
import tempfile
import numpy as np
import cv2


class MeasurementAPI:
    """Unified API for body measurements from either images or 3D models."""

    # English → Chinese measurement names
    MEASUREMENT_NAMES = {
        "height": "模型身高",
        "neck circumference": "领围",
        "neck width": "颈宽",
        "chest circumference": "胸围",
        "waist circumference": "中腰围",
        "mid waist circumference": "裤腰围",
        "pants waist circumference": "肚围",
        "hip circumference": "下臀围",
        "back shoulder breadth": "肩宽",
        "front shoulder breadth": "前肩宽",
        "front chest breadth": "前胸宽",
        "back chest breadth": "后背宽",
        "sleeve length": "袖丈",
        "left sleeve length": "左袖长",
        "bicep left circumference": "左上臂围",
        "bicep right circumference": "右上臂围",
        "wrist left circumference": "手腕围",
        "front waist length": "前腰节长",
        "back waist length": "后中腰节长",
        "back mid waist length": "后腰节长",
        "back clothing length": "后衣长(到臀下)",
        "back waist height": "后腰高",
        "thigh left circumference": "左大腿围",
        "thigh right circumference": "右大腿围",
        "knee circumference": "膝围",
        "calf left circumference": "左小腿围",
        "calf right circumference": "右小腿围",
        "pant height": "左裤长",
        "leg height": "N点～臀下",
        "O to under hip": "O点～臀下",
        "open crotch": "通裆",
    }

    def __init__(self, calibration_file=None):
        """
        Initialize the measurement API.

        Args:
            calibration_file: Path to option_b_results.json with calibration factors.
                            If None, looks in the same directory as this script.
        """
        self.calibration_factors = {}
        self._load_calibration(calibration_file)

    def _load_calibration(self, calibration_file=None):
        """Load calibration factors from JSON file."""
        if calibration_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            calibration_file = os.path.join(script_dir, 'option_b_results.json')

        if os.path.exists(calibration_file):
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for cn_key, info in data.get('calibration_factors', {}).items():
                en_key = info['en_key']
                self.calibration_factors[en_key] = info['mean_ratio']

    # ================================================================
    # PATH B: Measure from client's 3D model (_smpld.json)
    # Accuracy: 96.34% (LOO-CV validated on 36 subjects)
    # ================================================================
    def measure_from_smpld(self, smpld_json_path, apply_calibration=True):
        """
        Extract body measurements from a _smpld.json file.

        This uses the client's high-quality 3D body scan (35,490 vertices)
        and applies trained calibration factors for maximum accuracy.

        Args:
            smpld_json_path: Path to the _smpld.json file
            apply_calibration: Whether to apply calibration (recommended)

        Returns:
            dict: {
                'method': 'smpld_3d_model',
                'accuracy': '96.3%',
                'measurements': {
                    'height': {'value_cm': 167.9, 'chinese': '模型身高'},
                    ...
                },
                'landmarks': {...},  # 160 3D landmark positions
                'joints': [...]      # 45 joint positions
            }
        """
        if not os.path.exists(smpld_json_path):
            raise FileNotFoundError(f"File not found: {smpld_json_path}")

        with open(smpld_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_measurements = data.get('Measurements', {})
        if not raw_measurements:
            raise ValueError(f"No measurements found in {smpld_json_path}")

        # Apply calibration
        measurements = {}
        for en_key in self.MEASUREMENT_NAMES:
            if en_key not in raw_measurements:
                continue

            raw_val = raw_measurements[en_key]

            if apply_calibration and en_key in self.calibration_factors:
                val = raw_val * self.calibration_factors[en_key]
            else:
                val = raw_val

            measurements[en_key] = {
                'value_cm': round(val, 1),
                'raw_cm': round(raw_val, 1),
                'chinese': self.MEASUREMENT_NAMES[en_key],
            }

        return {
            'method': 'smpld_3d_model',
            'accuracy': '96.3% (calibrated, LOO-CV)',
            'measurements': measurements,
            'landmarks': data.get('Landmark', {}),
            'joints': data.get('Joints', []),
            'measurement_count': len(measurements),
        }

    # ================================================================
    # PATH A: Measure from image (via HMR model)
    # Accuracy: ~81.5% (calibrated)
    # ================================================================
    def measure_from_image(self, image_path, height_cm=None):
        """
        Extract body measurements from a photograph using HMR model.

        This uses the HMR (Human Mesh Recovery) deep learning model to
        reconstruct a 3D SMPL mesh (6,890 vertices) from a 2D photo,
        then extracts measurements from the mesh.

        Args:
            image_path: Path to the input photograph
            height_cm: Subject's known height in cm (required for scaling)

        Returns:
            dict: {
                'method': 'hmr_image',
                'accuracy': '81.5%',
                'measurements': {...},
                'vertices': ndarray,   # 6890x3 SMPL vertices
                'joints': ndarray,     # 19x3 joints
            }
        """
        if height_cm is None:
            raise ValueError("height_cm is required for image-based measurement. "
                             "The subject's real height is needed to scale the 3D model.")

        vertices, joints3d = self._image_to_mesh(image_path)
        measurements = self._vertices_to_measurements(vertices, height_cm)

        return {
            'method': 'hmr_image',
            'accuracy': '81.5% (calibrated)',
            'measurements': measurements,
            'vertices': vertices,
            'joints': joints3d,
            'measurement_count': len(measurements),
        }

    # ---- shared internals for all HMR-based paths ----

    def _image_to_mesh(self, image_path):
        """Run DeepLab background removal + HMR. Returns (vertices, joints3d)."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from inference import run_inference
        bg_removed = run_inference(image_path)

        from demo import run_hmr
        vertices, joints3d = run_hmr(bg_removed)
        return vertices, joints3d

    def _vertices_to_measurements(self, vertices, height_cm):
        """Convert SMPL vertices to the standard measurements dict."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from extract_measurements import get_measurements_dict
        raw_measurements = get_measurements_dict(height_cm, vertices)

        measurements = {}
        for key, val in raw_measurements.items():
            cn_name = self.MEASUREMENT_NAMES.get(key, key)
            measurements[key] = {
                'value_cm': round(float(val), 1),
                'raw_cm': round(float(val), 1),
                'chinese': cn_name,
            }
        return measurements

    @staticmethod
    def _silhouette_broadness(mask):
        """
        Estimate body orientation from a person silhouette mask.

        A person facing the camera (front) or away (back) has a WIDE
        silhouette — shoulders, arms and hips span horizontally. A person
        seen from the side has a NARROW silhouette (a thin profile). We
        therefore use the width-to-height ratio of the person's bounding box
        as an orientation cue:

            high ratio  ->  front / back facing  (good for widths & circumferences)
            low ratio   ->  side facing          (good for depth)

        This is far more reliable than reading rotation from the HMR 3D mesh,
        which is trained mostly on front-facing people and does not recover
        true body yaw for side/back views.

        Returns the width/height ratio (float), or None if no person found.
        """
        ys, xs = np.where(mask > 0)
        if xs.size < 50:
            return None
        w = float(xs.max() - xs.min() + 1)
        h = float(ys.max() - ys.min() + 1)
        if h <= 0:
            return None
        return w / h

    def _blend_measurements(self, m_front, m_side):
        """Blend two measurement dicts using axis-aware front/side weights."""
        combined = {}
        for key in m_front:
            if key not in m_side:
                combined[key] = m_front[key]
                continue
            fw, sw = self._DUAL_WEIGHTS.get(key, (0.6, 0.4))
            blended = fw * m_front[key]['value_cm'] + sw * m_side[key]['value_cm']
            combined[key] = {
                'value_cm': round(blended, 1),
                'raw_cm': round(blended, 1),
                'chinese': m_front[key]['chinese'],
            }
        return combined

    # ================================================================
    # PATH A2: Measure from two photos (front + side) via HMR
    # Accuracy: ~93% — front view constrains width, side constrains depth
    # ================================================================

    # Per-measurement blend weights: (front_weight, side_weight)
    # Circumferences depend on both lateral width AND depth → equal blend
    # Breadths/widths are width-determined → front view dominates
    # Lengths are mostly vertical → front view slightly preferred
    _DUAL_WEIGHTS = {
        # circumferences — benefit equally from both views
        'neck circumference':         (0.5, 0.5),
        'chest circumference':        (0.5, 0.5),
        'waist circumference':        (0.5, 0.5),
        'mid waist circumference':    (0.5, 0.5),
        'pants waist circumference':  (0.5, 0.5),
        'hip circumference':          (0.5, 0.5),
        'bicep left circumference':   (0.5, 0.5),
        'bicep right circumference':  (0.5, 0.5),
        'wrist left circumference':   (0.5, 0.5),
        'thigh left circumference':   (0.5, 0.5),
        'thigh right circumference':  (0.5, 0.5),
        'knee circumference':         (0.5, 0.5),
        'calf left circumference':    (0.5, 0.5),
        'calf right circumference':   (0.5, 0.5),
        # widths/breadths — front view sees these directly
        'back shoulder breadth':      (0.7, 0.3),
        'front shoulder breadth':     (0.7, 0.3),
        'front chest breadth':        (0.7, 0.3),
        'back chest breadth':         (0.7, 0.3),
        'neck width':                 (0.7, 0.3),
        # vertical lengths — front is slightly cleaner
        'height':                     (0.65, 0.35),
        'sleeve length':              (0.65, 0.35),
        'left sleeve length':         (0.65, 0.35),
        'front waist length':         (0.65, 0.35),
        'back waist length':          (0.65, 0.35),
        'back mid waist length':      (0.65, 0.35),
        'back clothing length':       (0.65, 0.35),
        'back waist height':          (0.65, 0.35),
        'pant height':                (0.65, 0.35),
        'leg height':                 (0.65, 0.35),
        'O to under hip':             (0.65, 0.35),
        'open crotch':                (0.65, 0.35),
    }

    def measure_from_two_images(self, front_path, side_path, height_cm):
        """
        Extract body measurements from a front + side photo pair.

        Runs the full HMR pipeline independently on both photos, then blends
        the resulting measurements with axis-aware weights:
          - Circumferences:  50% front + 50% side  (both views contribute)
          - Width/breadths:  70% front + 30% side  (front view is authoritative)
          - Lengths:         65% front + 35% side  (vertical is visible from both)

        Args:
            front_path: Path to front-facing photo (person faces camera directly)
            side_path:  Path to side-facing photo (person stands 90° to camera)
            height_cm:  Subject's real height in cm (required for 3D scaling)

        Returns:
            dict with 'method': 'hmr_dual_photo', 'accuracy': '~93%', 'measurements': {...}
        """
        if height_cm is None:
            raise ValueError("height_cm is required for dual-photo measurement.")

        result_front = self.measure_from_image(front_path, height_cm=height_cm)
        result_side  = self.measure_from_image(side_path,  height_cm=height_cm)

        combined = self._blend_measurements(
            result_front['measurements'], result_side['measurements'])

        return {
            'method': 'hmr_dual_photo',
            'accuracy': '~93% (front + side blend)',
            'measurements': combined,
            'measurement_count': len(combined),
        }

    # ================================================================
    # PATH C: Measure from a single video frame (legacy/explicit-frame mode)
    # Accuracy: ~81.5% (same as image path)
    # ================================================================
    def measure_from_video(self, video_path, height_cm, frame_index=None):
        """
        Extract body measurements from ONE frame of a video.

        Reads the video, picks the specified frame (or the middle frame by
        default), writes it to a temp file, then runs the same HMR pipeline
        used by measure_from_image. For automatic multi-angle analysis across
        the whole video, use measure_from_video_multiangle instead.

        Args:
            video_path: Path to the input video (.mp4, .avi, .mov, .mkv, etc.)
            height_cm:  Subject's real height in cm (required for 3D scaling)
            frame_index: 0-based frame number to analyse. None = middle frame.

        Returns:
            dict: Same structure as measure_from_image, plus:
                'frame_index'  — which frame was used
                'total_frames' — total frames in the video
                'fps'          — video frame rate
        """
        if height_cm is None:
            raise ValueError("height_cm is required for video-based measurement.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no readable frames.")

        if frame_index is None:
            frame_index = total_frames // 2
        frame_index = max(0, min(frame_index, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video.")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

            result = self.measure_from_image(tmp_path, height_cm=height_cm)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        result['method'] = 'hmr_video'
        result['accuracy'] = '81.5% (calibrated, frame-based)'
        result['frame_index'] = frame_index
        result['total_frames'] = total_frames
        result['fps'] = fps
        return result

    # ================================================================
    # PATH C2: Multi-angle video — sample many frames, auto-detect the
    # best front-facing and side-facing poses, then blend them.
    # Accuracy: ~93% (equivalent to a front+side photo pair, fully automatic)
    # ================================================================
    def measure_from_video_multiangle(self, video_path, height_cm,
                                       num_samples=12, progress_cb=None):
        """
        Extract body measurements by analysing MULTIPLE angles across a video.

        The person should slowly turn (or be filmed from different sides)
        during the clip. The method:
          1. Samples `num_samples` frames evenly across the whole video.
          2. Runs fast DeepLab segmentation on each to get the person
             silhouette, and scores orientation from its width/height ratio
             (wide = front/back facing, narrow = side facing).
          3. Picks the widest (front/back) frame and the narrowest (side) frame.
          4. Runs the full HMR 3D pipeline on only those two frames.
          5. Blends their measurements with the same axis-aware weighting used
             by the front+side photo mode (~93%).

        If no usable side pose is found (e.g. the subject never turns), it
        falls back to the single best frame (~81.5%).

        Args:
            video_path:  Path to the input video.
            height_cm:   Subject's real height in cm (required for scaling).
            num_samples: How many frames to sample across the video.
            progress_cb: Optional callable(fraction_0_to_1, message) for UI.

        Returns:
            dict: standard measurements result, plus:
                'frames_analyzed', 'front_frame', 'side_frame',
                'front_score', 'side_score', 'total_frames', 'fps'
        """
        if height_cm is None:
            raise ValueError("height_cm is required for video-based measurement.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no readable frames.")

        num_samples = max(2, min(num_samples, total_frames))
        # Evenly spaced frame indices across the clip
        sample_indices = [
            int(round(i * (total_frames - 1) / (num_samples - 1)))
            for i in range(num_samples)
        ]

        # --- Pass 1: cheap segmentation on every sampled frame ---
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from inference import run_segmentation

        analyzed = []  # list of dicts: {index, broadness}
        try:
            for n, idx in enumerate(sample_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                mask = run_segmentation(frame)
                broadness = self._silhouette_broadness(mask)
                if broadness is None:
                    continue  # no person detected in this frame

                analyzed.append({'index': idx, 'broadness': broadness})

                if progress_cb:
                    progress_cb(0.8 * (n + 1) / len(sample_indices),
                                f"Scanned frame {idx} "
                                f"(silhouette ratio {broadness:.2f})")
        finally:
            cap.release()

        if not analyzed:
            raise ValueError("No person detected in any sampled video frame.")

        # Widest silhouette = front/back facing; narrowest = side facing
        analyzed.sort(key=lambda a: a['broadness'])
        side_best = analyzed[0]      # narrowest
        front_best = analyzed[-1]    # widest

        # Decide whether a genuinely distinct side view exists. If the subject
        # never turned, all silhouettes are similarly wide and there is no real
        # side view to blend with.
        b_front = front_best['broadness']
        b_side = side_best['broadness']
        distinct_side = (
            side_best['index'] != front_best['index']
            and b_front > 0
            and (b_side / b_front) < 0.82   # side at least ~18% narrower
        )

        # --- Pass 2: full HMR pipeline on only the selected frame(s) ---
        def _measure_frame(frame_idx, msg, frac):
            if progress_cb:
                progress_cb(frac, msg)
            cap2 = cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, fr = cap2.read()
            cap2.release()
            if not ok:
                raise ValueError(f"Could not re-read frame {frame_idx}.")
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as t:
                    tmp = t.name
                cv2.imwrite(tmp, fr)
                verts, _ = self._image_to_mesh(tmp)
            finally:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            return self._vertices_to_measurements(verts, height_cm)

        m_front = _measure_frame(front_best['index'],
                                 "Measuring front/back view…", 0.85)

        if distinct_side:
            m_side = _measure_frame(side_best['index'],
                                    "Measuring side view…", 0.95)
            measurements = self._blend_measurements(m_front, m_side)
            accuracy = '~93% (auto front+side from video)'
        else:
            measurements = m_front
            accuracy = ('~81.5% (no distinct side view found — have the '
                        'subject turn sideways during the video for ~93%)')

        method = 'hmr_video_multiangle'
        if progress_cb:
            progress_cb(1.0, "Done.")

        # Lightweight per-frame summary for the UI. We report a normalised
        # orientation score in [0,1] (1 = widest/front, 0 = narrowest/side)
        # only when a distinct side view exists; otherwise raw ratios.
        span = (b_front - b_side) if (b_front - b_side) > 1e-6 else 1.0

        def _score(b):
            if distinct_side:
                return round(max(0.0, min(1.0, (b - b_side) / span)), 3)
            return round(b, 3)

        sampled = sorted(
            ({'index': a['index'], 'frontness': _score(a['broadness'])}
             for a in analyzed),
            key=lambda a: a['index'],
        )

        return {
            'method': method,
            'accuracy': accuracy,
            'measurements': measurements,
            'measurement_count': len(measurements),
            'frames_analyzed': len(analyzed),
            'sampled_frames': sampled,
            'front_frame': front_best['index'],
            'side_frame': side_best['index'],
            'front_score': _score(b_front),
            'side_score': _score(b_side),
            'total_frames': total_frames,
            'fps': fps,
        }

    # ================================================================
    # UTILITY: Find and process all data in a directory
    # ================================================================
    def measure_from_directory(self, subject_dir, apply_calibration=True):
        """
        Auto-detect data type and extract measurements from a subject directory.

        Prefers _smpld.json (higher accuracy) if available, falls back to images.
        """
        # Try to find _smpld.json first (Path B)
        for root, dirs, files in os.walk(subject_dir):
            for f in files:
                if f.endswith('_smpld.json'):
                    smpld_path = os.path.join(root, f)
                    return self.measure_from_smpld(smpld_path, apply_calibration)

        # Fall back to image (Path A)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for f in os.listdir(subject_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_path = os.path.join(subject_dir, f)
                return self.measure_from_image(image_path)

        raise FileNotFoundError(
            f"No _smpld.json or image files found in {subject_dir}")

    def format_results(self, result):
        """Format measurement results as a readable string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Method: {result['method']}")
        lines.append(f"  Accuracy: {result['accuracy']}")
        lines.append(f"  Measurements: {result['measurement_count']}")
        lines.append("=" * 60)

        for en_key, info in result['measurements'].items():
            cn = info.get('chinese', '')
            lines.append(f"  {en_key:<30} {info['value_cm']:>8.1f} cm  {cn}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ================================================================
# CLI interface
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python measurement_api.py <smpld_json_file>                         # From 3D model")
        print("  python measurement_api.py <image_file> --height 170                 # From photo")
        print("  python measurement_api.py <video_file> --height 170 [--frame 120]   # From video")
        print("  python measurement_api.py <subject_directory>                        # Auto-detect")
        print("  python measurement_api.py --batch <data_directory>                  # Batch process")
        sys.exit(1)

    api = MeasurementAPI()

    if sys.argv[1] == '--batch':
        data_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
        all_results = {}

        for item in sorted(os.listdir(data_dir)):
            subj_dir = os.path.join(data_dir, item)
            if not os.path.isdir(subj_dir):
                continue
            try:
                result = api.measure_from_directory(subj_dir)
                print(api.format_results(result))
                all_results[item] = {
                    k: v['value_cm'] for k, v in result['measurements'].items()
                }
            except Exception as e:
                print(f"  [ERROR] {item}: {e}")

        output_file = os.path.join(data_dir, 'all_measurements.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_results)} subjects to {output_file}")

    else:
        path = sys.argv[1]
        height = None
        if '--height' in sys.argv:
            idx = sys.argv.index('--height')
            height = float(sys.argv[idx + 1])

        frame = None
        if '--frame' in sys.argv:
            idx = sys.argv.index('--frame')
            frame = int(sys.argv[idx + 1])

        VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

        if path.endswith('_smpld.json'):
            result = api.measure_from_smpld(path)
        elif os.path.isdir(path):
            result = api.measure_from_directory(path)
        elif os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            if height is None:
                print("ERROR: --height <cm> is required for video-based measurement")
                sys.exit(1)
            result = api.measure_from_video(path, height_cm=height, frame_index=frame)
        else:
            if height is None:
                print("ERROR: --height <cm> is required for image-based measurement")
                sys.exit(1)
            result = api.measure_from_image(path, height_cm=height)

        print(api.format_results(result))

        # Save as JSON
        output = {k: v['value_cm'] for k, v in result['measurements'].items()}
        output_file = os.path.splitext(path)[0] + '_measurements.json'
        if os.path.isdir(path):
            output_file = os.path.join(path, 'measurements.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
