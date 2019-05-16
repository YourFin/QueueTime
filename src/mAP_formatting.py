from file_management import QUEUETIME_DIR
from annotations import get_image_annotations

MAP_GROUND_TRUTH_DIR = QUEUETIME_DIR + '/data/mAP/ground_truth'
MAP_CLASSIFIED_DIR = QUEUETIME_DIR + '/data/mAP/classified'

def bbox_to_txt_line(ann):
    """
    Given an absolute annotation in the coco dataset, return the corresponding
    text line in txt file
    """
    bbox = ann['bbox']
    right = bbox[0] + bbox[2]
    bottom = bbox[1] + bbox[3]
    return "human %d %d %d %d\n" % (bbox[0], bbox[1], right, bottom)

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

def classified_write_anns_to_file(img_id, anns):
    """
    Write the annotations resulting from running the neural network to the file
    $MAP_CLASSIFIED_DIR/$img_id.txt
    """
    fname = '%s/%012d.%s' % (MAP_CLASSIFIED_DIR, img_id, 'txt')
    write_anns_to_file(anns, fname)
