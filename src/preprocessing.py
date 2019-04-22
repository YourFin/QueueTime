import numpy as np
from math import ceil

# Procedure:
#  pad_image
# Purpose:
#  To pad an image in pre-processing to a square aspect ratio
# Parameters:
#  img: numpy[int][int][int] - A numpy array representing a color image
# Produces:
#  output: numpy[int][int][int] - A numpy array representing a color image
# Preconditions:
#  No additional
# Postconditions:
#  dim_size(output, 0) == dim_size(output, 1)
def pad_image(img):
    (X_size, Y_size, _) = img.shape
    if X_size > Y_size:
        dif = X_size - Y_size
        return np.pad(img, ((0,0),(0,dif),(0,0)), 'constant')  # Default to 0
    elif X_size == Y_size:
        return img
    else:
        dif = Y_size - X_size
        return np.pad(img, ((0,dif),(0,0),(0,0)), 'constant')  # Default to 0

# Procedure:
#  pad_mask
# Purpose:
#  To pad an mask in pre-processing to a square aspect ratio
# Parameters:
#  mask: numpy[int][int] - A numpy array representing a 2d mask
# Produces:
#  output: numpy[int][int] - A numpy array representing a 2d mask
# Preconditions:
#  No additional
# Postconditions:
#  dim_size(output, 0) == dim_size(output, 1)
def pad_image(mask):
    (X_size, Y_size) = mask.shape
    if X_size > Y_size:
        dif = X_size - Y_size
        return np.pad(mask, ((0,0),(0,dif)), 'constant')  # Default to 0
    elif X_size == Y_size:
        return mask
    else:
        dif = Y_size - X_size
        return np.pad(mask, ((0,dif),(0,0)), 'constant')  # Default to 0


# Procedure:
#  gen_training_tensors
# Purpose:
#  To generate the ground truth corresponding to a given image.
# Parameters:
#  coco: COCO - a coco instance to pull annotation information from
#  bounding_box_count: int - number of bounding boxes per cell; known as B in YOLO paper.
#  cell_width: int - the width in pixels of a cell in the image.
#  cell_height: int - the height in pixels of a cell in the image.
#  img: pycocotools.img - image metadata object for the image
# Produces:
#  output: tensor[double] - A tensor of training data
# Preconditions:
#  coco is initialized with valid data
#  cell_width < width of image
#  cell_height < height of image
def gen_training_tensors(coco, bounding_box_count, cell_width, cell_height, img):
    person_cat_ids = coco.getCatIds(catNms=['person'])
    annotation_ids = coco.getAnnIds(imgIds=[img['id']], catIds=person_cat_ids)
    annotations = coco.loadAnns(annotation_ids)

    cell_x_conut = ceil(img.shape[0] / cell_size)
    cell_y_conut = ceil(img.shape[1] / cell_size)

    # 5 parameters to each bounding box: Probability, X pos, Y pos, Width, Height
    training_data = np.zeros((cell_x_count, cell_y_count, bounding_box_count * 5))


# Returns a generator of tuples: (img, training tensor)
def get_training_data_generator(cell_rows, cell_columns, bounding_box_count, image_width, image_height):
    pass
