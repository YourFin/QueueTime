from file_management import ANNOTATION_FILE, IMAGES_DIR

# Hardcoded parameters
CATEGORY_NAMES = ['person']  # person category should be 1
SUPERCATEGORY_NAMES = []

# Procedure:
#  download_imgs
# Purpose:
#  download the images in $image_ids
# Parameters:
#  coco: pycocotools.coco.coco - a coco dataset object
#  image_ids: [int] - a list of image ids to download
# Produces:
#  Side effects (file system)
# Preconditions:
#  $file_management.IMAGES_DIR exists as a directory
# Postconditions:
#  There exist files in $file_management.IMAGES_DIR corresponding to
#  each id in $image_ids
def download_imgs(coco, image_ids):
    coco.download(IMAGES_DIR, image_ids)

# Procedure:
#  download_some_imgs
# Purpose:
#  To download some images with people in them
# Parameters:
#  coco: pycocotools.coco.coco - a coco dataset object
#  image_count: int - the number of images to download
# Produces:
#  Side effects (file system)
# Preconditions:
#  image_count >= 1 || image_count == -1
#  $file_management.IMAGES_DIR exists as a directory
# Postconditions:
#  at least $image_count images exist in $file_management.IMAGES_DIR
#  if image_count > the number of images with people in them, the max
#   amount will be downloaded.
#  if image_count == -1, the maximum amount of images will be downloaded
def download_some_imgs(coco, image_count):
    category_ids = coco.getCatIds(CATEGORY_NAMES, SUPERCATEGORY_NAMES, [])
    image_ids = coco.getImgIds([],category_ids)
    image_ids = image_ids[:NUM_IMAGES]
    download_imgs(coco, image_ids)


if __name__ == '__main__':
    from pycocotools.coco import COCO

    NUM_IMAGES = 3  # -1 for all
    coco = COCO(ANNOTATION_FILE)
    download_some_imgs(coco, NUM_IMAGES)
