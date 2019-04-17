import os
import logging
from os.path import dirname
import re

QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__)))
DATASET_DIR = '%s/data/coco' % QUEUETIME_DIR
ANNOTATION_FILE = '%s/annotations/instances_train2017.json' % DATASET_DIR

# Procedure:
#  get_downloaded_ids
# Purpose:
#  return the ids of all downloaded images
# Parameters:
#  None
# Produces:
#  ids [int] - A list of integers
# Preconditions:
#  None
# Postconditions:
#  for each file in queuetime/data/coco/images/, $ids will contain the number
#  name of the file stripped of the .jpg
def get_downloaded_ids():
    image_extension = 'jpg'
    files = os.listdir(DATASET_DIR + '/images')
    # Verifies that it is an image
    img_pattern = re.compile('\d+\.jpg')
    ext_pattern = re.compile('\.jpg$')
    ids = []
    for file in files:
        if not img_pattern.match(file):
            logging.warning('Warning: \'%s\' in %s does not follow the standard image format in the coco dataset')
            continue

        ids.append(int(ext_pattern.sub('', file)))
    return ids
