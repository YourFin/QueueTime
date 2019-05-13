import json
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN

from os.path import dirname
import os

QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__)))
DATASET_DIR = '%s/data/coco' % QUEUETIME_DIR
ANNOTATION_FILE = '%s/annotations/instances_train2017.json' % DATASET_DIR

ROOT_DIR = dirname(os.path.abspath(__file__))
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Borrowed heavily from: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400``

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

def assert_dataset_downloaded():
    "Downloads the pre-trained coco model if need be"
    if not os.path.exists(COCO_MODEL_PATH):
        mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)


def evaluate_video(video_path):
    assert os.path.exists(video_path), "Video does not exist"
    vidstream = cv2.VideoCapture(video_path)




if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video-path", type=Path, help="Path to input video")
    ap.add_argument("-o", "--output", type=Path, help="Path to output file")
    args = ap.parse_args()
