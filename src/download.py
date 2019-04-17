from pycocotools.coco import COCO

# Hardcoded parameters
CATEGORY_NAMES = ['person']
SUPERCATEGORY_NAMES = []
NUM_IMAGES = 1000  # -1 for all

category_ids = COCO.getCatIds(CATEGORY_NAMES, SUPERCATEGORY_NAMES, [])
image_ids = COCO.getImgIds([],category_ids)

image_ids = image_ids[:NUM_IMAGES]

COCO.download('../data/datasets/COCO', image_ids)
