"""
HMR (Human Mesh Recovery) - 3D body mesh from a single image.

Reconstructs a 6,890-vertex SMPL mesh from a 2D photo, then extracts
body measurements. Requires the image to be roughly centered on the person.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import extract_measurements
import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel


def preprocess_image(img_path, json_path=None):
    img = img_path
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != 224:
            scale = (float(224) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               224)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, height, json_path=None):
    sess = tf.Session()
    model = RunModel(sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    extract_measurements.extract_measurements(height, verts[0])


# ── Importable API ────────────────────────────────────────────────────────────
# Lazy singleton so the heavy model is loaded only once per process.
_hmr_sess = None
_hmr_model = None


def _get_hmr_model():
    global _hmr_sess, _hmr_model
    if _hmr_model is None:
        print('Loading HMR model …')
        _hmr_sess = tf.Session()
        _hmr_model = RunModel(sess=_hmr_sess)
        print('HMR model loaded.')
    return _hmr_sess, _hmr_model


def run_hmr(bg_removed_image):
    """
    Run HMR inference on a background-removed numpy image (BGR uint8).

    Returns:
        vertices  — (6890, 3) float32 SMPL mesh vertices
        joints3d  — (19, 3)  float32 3D joint positions
    """
    _, model = _get_hmr_model()

    input_img, _, _ = preprocess_image(bg_removed_image)
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    return verts[0], joints3d[0]

