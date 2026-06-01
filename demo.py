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

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    extract_measurements.extract_measurements(height, verts[0])

