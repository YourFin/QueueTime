import numpy as np

def abs_ann_to_mask(mask_width, mask_height, bbox):
    """
    Converts a coco style $bbox list into a mask
    with size $mask_width by $mask_height

    mask_width and mask_height both need to be integers,
    bbox in the form [x_pos: int, y_pos: int, width: int, height: int]
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


def test_abs_ann_to_mask()
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
    ax.imshow(abs_ann_to_mask(640, 420, ann['bbox']))
    plt.figure(1)
    plot_annotations(img_ids[0], [ann])
