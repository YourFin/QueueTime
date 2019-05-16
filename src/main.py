#!/usr/bin/env python3 -i

from pycocotools.coco import COCO
from file_management import ANNOTATION_FILE, get_downloaded_ids
from annotations import get_image_annotations, plot_annotations
from datagenerator import CocoDataGenerator
from preprocessing import gen_training_tensor

#coco = COCO(ANNOTATION_FILE)
img_ids = get_downloaded_ids()
#anns = [get_image_annotations(coco, img) for img in img_ids]
#view_img = lambda index: plot_annotations(img_ids[index], anns[index])

test_tensor = lambda img_id: gen_training_tensor(coco, 1, 32, 32, img_id)

coco_datagenerator = CocoDataGenerator(
    640,
    640,
    32
)
