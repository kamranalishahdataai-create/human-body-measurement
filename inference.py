"""
Inference pipeline: Image → DeepLab segmentation → Background removal → HMR → Measurements.

Usage:
    python inference.py -i sample_data/input/image.jpg -ht 170
"""
import os
from io import BytesIO

import src.config
import tarfile
from six.moves import urllib

import numpy as np
from PIL import Image
import cv2, argparse
from demo import main
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		#"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
	A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
		  colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap

def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
	label: A 2D array with integer type, storing the segmentation label.

	Returns:
	result: A 2D array with floating type. The element of the array
	  is the color indexed by the corresponding element in the input label
	  to the PASCAL color map.

	Raises:
	ValueError: If label is not of rank 2 or its value is larger than color
	  map maximum entry.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]



_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEEPLAB_MODEL_NAME = 'xception_coco_voctrainval'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}

# Lazy-loaded singleton — avoids loading the model on every import
_deeplab_singleton = None


def _get_deeplab_model():
    global _deeplab_singleton
    if _deeplab_singleton is None:
        model_dir = os.path.join(_PROJECT_DIR, 'deeplab_model')
        os.makedirs(model_dir, exist_ok=True)
        tarball = _MODEL_URLS[_DEEPLAB_MODEL_NAME]
        download_path = os.path.join(model_dir, tarball)
        if not os.path.exists(download_path):
            print(f'Downloading DeepLab model to {download_path} …')
            urllib.request.urlretrieve(
                _DOWNLOAD_URL_PREFIX + tarball, download_path)
            print('Download complete.')
        print('Loading DeepLab model …')
        _deeplab_singleton = DeepLabModel(download_path)
        print('DeepLab model loaded.')
    return _deeplab_singleton


def run_inference(image_path):
    """
    Remove the background from an image using DeepLab segmentation.

    Accepts a file path (str) or a numpy BGR array.
    Returns a numpy array (BGR, uint8) with the background replaced by white.
    """
    model = _get_deeplab_model()

    if isinstance(image_path, np.ndarray):
        img_bgr = image_path
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.open(image_path)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    _, seg = model.run(pil_image)
    seg = cv2.resize(seg.astype(np.uint8), pil_image.size)
    mask = (255 * (seg == 15).astype(np.uint8))

    res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    return bg_removed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deeplab Segmentation')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to input image.')
    parser.add_argument('-ht', '--height', type=int, required=True,
                        help='Subject height in cm.')
    args = parser.parse_args()

    bg_removed = run_inference(args.input_dir)
    main(bg_removed, args.height, None)


