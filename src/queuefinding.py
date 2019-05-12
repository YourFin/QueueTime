import numpy as np

def abs_ann_to_mask(mask_width, mask_height, bbox):
    """
    Converts a coco style $bbox list into a mask
    with size $mask_width by $mask_height

    mask_width and mask_height both need to be integers,
    bbox in the form [x_pos: int, y_pos: int, width: int, height: int]
    """
    mask = np.zeros((mask_height, mask_width), np.float)
    mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
    return mask


if __name__ == '__main__':
    import matplotlib as plt
    from pycocotools.coco import COCO
    from file_management import ANNOTATION_FILE, get_downloaded_ids
    from annotations import get_image_annotations, plot_annotations
    from datagenerator import CocoDataGenerator
    from preprocessing import gen_training_tensor

    coco = COCO(ANNOTATION_FILE)
    img_ids = get_downloaded_ids()
    anns = [get_image_annotations(coco, img) for img in img_ids]
    #view_img = lambda index: plot_annotations(img_ids[index], anns[index])

    ann = anns[0][0]
    plt.figure(1)
    plot_annotations(image_ids[0], [ann])
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.imshow(abs_ann_to_mask(640, 640, ann['bbox']))
