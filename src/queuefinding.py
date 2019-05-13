import numpy as np
import cv2

def single_abs_ann_to_rect_mask(mask_width, mask_height, bbox):
    """
    Converts a single coco style $bbox list into a mask
    with size $mask_width by $mask_height

    mask_width and mask_height both need to be integers,
    bbox in the form [x_pos: int, y_pos: int, width: int, height: int]

    All indices in the rectangle will be 1, the rest 0

    returns $mask_height by $mask_width numpy array of type float
    """
    # Force cast to ints as these sometimes get loaded in as floats
    bbox = [int(item) for item in bbox]

    assert bbox[0] + bbox[2] < mask_width, 'bounding box not in the image'
    assert bbox[0] >= 0, 'bounding box not in the image'
    assert bbox[1] + bbox[3] < mask_width, 'bounding box not in the image'
    assert bbox[1] >= 0, 'bounding box not in the image'

    mask = np.zeros((mask_height, mask_width), np.float)
    mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
    return mask

def abs_anns_to_heatmap(mask_width, mask_height, anns):
    """
    Converts a coco style list of annotations into a location heatmap,
    of size $mask_width by $mask_height.

    $mask_width and $mask_height both need to be integers
    $anns needs to be in the form:
      [{'bbox': [x_pos: int, y_pos: int, width: int, height: int], ...}]
    containing all relevant bounding boxes

    Implementation:
    1. generate a numpy array of size ($mask_height, $mask_width).
    2. For each annotation rectangle, add 1 to every pixel in the rectangle.
    3. Normalize the image to the range [0,1] by dividing by the max value.
     - May be worth looking into doing this non-linearly
    4. Preform a gaussian blur on the heatmap

    returns $mask_height by $mask_width numpy array of type float, all values
    between 1 and 0.
    """
    mask = np.zeros((mask_height, mask_width), np.float)
    for ann in anns:
        mask += single_abs_ann_to_rect_mask(mask_width, mask_height, ann['bbox'])

    mask = mask / mask.max()
    return mask

def test_abs_anns_to_heatmap():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from pycocotools.coco import COCO
    from file_management import ANNOTATION_FILE, get_downloaded_ids
    from annotations import get_image_annotations, plot_annotations

    coco = COCO(ANNOTATION_FILE)
    img_ids = get_downloaded_ids()
    anns = [get_image_annotations(coco, img) for img in img_ids]

    plt.figure(0)
    ann = anns[0]
    heatmap = abs_anns_to_heatmap(640, 640, ann)
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.imshow(heatmap) #, cmap=cm.jet)
    plt.figure(1)
    plot_annotations(img_ids[0], ann)


def test_single_abs_ann_to_rect_mask():
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from file_management import ANNOTATION_FILE, get_downloaded_ids
    from annotations import get_image_annotations, plot_annotations

    coco = COCO(ANNOTATION_FILE)
    img_ids = get_downloaded_ids()
    anns = [get_image_annotations(coco, img) for img in img_ids]
    #view_img = lambda index: plot_annotations(img_ids[index], anns[index])

    ann = anns[0][0]
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.imshow(single_abs_ann_to_rect_mask(640, 420, ann['bbox']))
    plt.figure(1)
    plot_annotations(img_ids[0], [ann])
