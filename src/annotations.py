import matplotlib.patches as patches
import matplotlib.pyplot as plt
from file_management import get_image

# Procedure:
#  plot_annotations
# Purpose:
#  display a given set of bounding boxes on an image
# Parameters:
#  img_id: int - The id of the image to display
#  annotations: [dict] - list of annotations to display
#  color: string = r - the matplotlib color to plot the boxes in
# Produces:
#  Side effects: displays the given image with matplotlib
# Preconditions:
#  annotations are dicts matching the annotation format of the
#   coco dataset, primarily containing the bbox key
#  img_id has been downloaded
# Postconditions:
#  The image corresponding to $img_id is displayed with $annotations in $color
def plot_annotations(img_id, annotations, color='r'):
    img = get_image(img_id)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for ann in annotations:
        rect_patch = patches.Rectangle(
            (ann['bbox'][0], ann['bbox'][1]),
            ann['bbox'][2],
            ann['bbox'][3],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect_patch)
    plt.show()

# Procedure:
#  get_image_annotations
# Purpose:
#  To get the annotations corresponding to a given image
# Parameters:
#  coco: pycocotools.coco.coco
#  img_id: int - id of the image to look up
# Produces:
#  output: [dict(str, object)] - annotation data for a given object
# Preconditions:
#  img_id is valid in $coco
# Postconditions:
#  All annotations in $coco of humans are returned in output
def get_image_annotations(coco, img_id):
    img = coco.loadImgs([img_id])[0]
    person_cat_ids = coco.getCatIds(catNms=['person'])
    annotation_ids = coco.getAnnIds(imgIds=[img['id']], catIds=person_cat_ids, iscrowd=False)
    return coco.loadAnns(annotation_ids)
