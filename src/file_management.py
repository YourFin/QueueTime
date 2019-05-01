import os
import logging
from os.path import dirname
import re
from matplotlib.image import imread

QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__)))
DATASET_DIR = '%s/data/coco' % QUEUETIME_DIR
ANNOTATION_FILE = '%s/annotations/instances_train2017.json' % DATASET_DIR
IMAGES_DIR = DATASET_DIR + '/images/'

IMAGE_EXTENSION = 'jpg'


# Procedure:
#  get_downloaded_ids
# Purpose:
#  return the ids of all downloaded images
# Parameters:
#  None
# Produces:
#  ids: [int] - A list of integers
# Preconditions:
#  None
# Postconditions:
#  for each file in queuetime/data/coco/images/, $ids will contain the number
#  name of the file stripped of the .jpg
def get_downloaded_ids():
    files = os.listdir(IMAGES_DIR)
    # Verifies that it is an image
    img_pattern = re.compile('\d+\.' + IMAGE_EXTENSION)
    ext_pattern = re.compile('\.' + IMAGE_EXTENSION + '$')
    ids = []
    for file in files:
        if not img_pattern.match(file):
            logging.warning('Warning: \'%s\' in %s does not follow the standard image format in the coco dataset')
            continue

        ids.append(int(ext_pattern.sub('', file)))
    return ids

# Procedure:
#  get_image
# Purpose:
#  Return the image array with the given id
# Parameters:
#  id: int - the id of the image to be loaded
# Produces:
#  img: numpy[int][int][int] - the image in the file
# Preconditions:
#  A picture with the given id exists
# Postconditions:
#  Trivial
def get_image(id):
    try:
        img_array = imread('%s%012d.%s' % (IMAGES_DIR, id, IMAGE_EXTENSION))
    except FileNotFoundError:
        raise FileNotFoundError(
            'The file corresponding to %d does not exist; ' % id +
            'please download it with download.download_imgs'
        )
    # Remove alpha channel if present
    return img_array[:,:,:3]
