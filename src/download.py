from pycocotools.coco import COCO
from file_management import ANNOTATION_FILE

# Hardcoded parameters
CATEGORY_NAMES = ['person']
SUPERCATEGORY_NAMES = []
NUM_IMAGES = 3  # -1 for all

coco = COCO(ANNOTATION_FILE)

category_ids = coco.getCatIds(CATEGORY_NAMES, SUPERCATEGORY_NAMES, [])
image_ids = coco.getImgIds([],category_ids)

image_ids = image_ids[:NUM_IMAGES]

coco.download('../data/coco/images/', image_ids)
