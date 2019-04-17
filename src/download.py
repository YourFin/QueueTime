from pycocotools.coco import COCO
import os
from os.path import dirname

# Hardcoded parameters
CATEGORY_NAMES = ['person']
SUPERCATEGORY_NAMES = []
NUM_IMAGES = 3  # -1 for all
QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__))) + ""
DATASET_DIR = '%s/data/coco' % QUEUETIME_DIR
ANNOTATION_FILE = '%s/annotations/instances_train2017.json' % DATASET_DIR

coco = COCO(ANNOTATION_FILE)

category_ids = coco.getCatIds(CATEGORY_NAMES, SUPERCATEGORY_NAMES, [])
image_ids = coco.getImgIds([],category_ids)

image_ids = image_ids[:NUM_IMAGES]

coco.download('../data/COCO', image_ids)
