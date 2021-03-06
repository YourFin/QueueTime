import matplotlib.patches as patches
import matplotlib.pyplot as plt
from file_management import get_image
from math import floor

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
#  If the 'color' key exist in anns, that is used as the color instead of
#   $color
# Practica:
#  As an example of coloring by score:
#  # Assume $output is the output of the NN
#  anns = cnn_y_to_absolute(CELLL_WIDTH, CELL_HEIGHT, output)
#  for ann in anns:
#    ann['color'] = plt.cm.jet(ann['score'])
#  plot_annotations(img_id, anns)
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
            edgecolor=ann.get('color', color),
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

# Procedure:
#  cnn_y_to_absolute
# Purpose:
#  To generate absolute coordinate annotation data from the neural network output
# Paramaters:
#  cell_width: int - width of cells in pixels
#  cell_height: int - height of cells in pixels
#  output_data: numpy[NxMxB*5] - output data from neural network
# Produces:
#  bounding_boxes: [{'bbox': [int, int, int, int], 'score': int}] - list of bounding boxes as in coco dataset
# Preconditions:
#  no additional
# Postconditions:
#  each bbox entry will be of the form (ul_x_pos, ul_y_pos, width, height)
#  Algorithm:
#   Transform values back to pixels:
#    width *= cell_width
#    height *= cell_height
#   Transform coordinates back to coco style list as (ul_x_pos, ul_y_pos,width, height)
#   score key contains calculated score of the bounding box
#   x_pos and y_pos are representing the upper left corner of the box
def cnn_y_to_absolute(cell_width, cell_height, output_data):
    # Need to lift into upper file
    POS_OBJ_SCORE = 0
    POS_BOX_CENTER_X = 1
    POS_BOX_CENTER_Y = 2
    POS_BOX_WIDTH = 3
    POS_BOX_HEIGHT = 4
    (y_cells, x_cells, channels) = output_data.shape

    bounding_boxes = []

    for x_cell in range(x_cells):
        for y_cell in range(y_cells):
            for box_num in range(floor(channels / 5)):
                rel_width = output_data[y_cell, x_cell, box_num * 5 + POS_BOX_WIDTH]
                rel_height = output_data[y_cell, x_cell, box_num * 5 + POS_BOX_HEIGHT]

                absolute_width = rel_width * cell_width * 10 #hard code
                absolute_height = rel_height * cell_height * 10 #hard code

                rel_box_center_x = output_data[y_cell, x_cell, box_num * 5 + POS_BOX_CENTER_X]
                rel_box_center_y = output_data[y_cell, x_cell, box_num * 5 + POS_BOX_CENTER_Y]

                cell_box_center_x = rel_box_center_x * cell_width
                cell_box_center_y = rel_box_center_y * cell_height

                cell_box_ul_x = cell_box_center_x - absolute_width / 2
                cell_box_ul_y = cell_box_center_y - absolute_height / 2

                abs_box_ul_x = cell_box_ul_x + (x_cell * cell_width)
                abs_box_ul_y = cell_box_ul_y + (y_cell * cell_height)

                score = output_data[y_cell, x_cell, box_num * 5 + POS_OBJ_SCORE]

                bounding_box = {
                    'bbox': [abs_box_ul_x, abs_box_ul_y, absolute_width, absolute_height],
                    'score': score
                }
                bounding_boxes.append(bounding_box)

    return bounding_boxes
