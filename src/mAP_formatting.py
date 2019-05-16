#!/usr/bin/env python3
# If this file is called as a script, it will generate mAP files for all
# image ids listed in the arguments
#####

from file_management import QUEUETIME_DIR
from annotations import get_image_annotations

MAP_GROUND_TRUTH_DIR = QUEUETIME_DIR + '/mAP/input/ground-truth'
MAP_CLASSIFIED_DIR   = QUEUETIME_DIR + '/mAP/input/detection-results'

def bbox_to_txt_line(ann):
    """
    Given an absolute annotation in the coco dataset, return the corresponding
    text line in txt file
    """
    bbox = ann['bbox']
    right = bbox[0] + bbox[2]
    bottom = bbox[1] + bbox[3]
    if 'normalized_score' in ann:
        confidence = ann['normalized_score']
    else:
        # If there isn't a normalized score, this is coming from the ground truth
        confidence = 1
    return "human %.6f %d %d %d %d\n" % (confidence, bbox[0], bbox[1], right, bottom)

def write_anns_to_file(anns, fname):
    """
    Given a list of annotations $anns, write them to the file $fname
    """
    with open(fname, 'w') as fhandle:
        for ann in anns:
            fhandle.write(bbox_to_txt_line(ann))

def coco_write_anns_to_file(coco, img_id):
    """
    Given an image id $img_id, write out the annotations in the
    mAP format to $MAP_GROUND_TRUTH_DIR/$image_id.txt
    """
    anns = get_image_annotations(coco, img_id)
    fname = '%s/%012d.%s' % (MAP_GROUND_TRUTH_DIR, img_id, 'txt')
    write_anns_to_file(anns, fname)

def classified_write_anns_to_file(anns, img_id):
    """
    Write the annotations resulting from running the neural network to the file
    $MAP_CLASSIFIED_DIR/$img_id.txt
    """
    fname = '%s/%012d.%s' % (MAP_CLASSIFIED_DIR, img_id, 'txt')
    write_anns_to_file(anns, fname)

if __name__ == '__main__':
    from file_management import ANNOTATION_FILE, get_downloaded_ids
    from pycocotools.coco import COCO
    from sys import argv

    coco = COCO(ANNOTATION_FILE)

    assert os.path.exists(MAP_CLASSIFIED_DIR), """
                                               mAP dirs do not exist, run:
                                               git submodule init
                                               git submodule update
                                               """

    for img_id in argv:
        try:
            coco_write_anns_to_file(coco, int(img_id))
        except ValueError:
            print(img_id, "is not valid")
            exit(1)
