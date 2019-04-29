import logging
import numpy as np
from math import ceil
import file_management

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
    annotation_ids = coco.getAnnIds(imgIds=[img['id']], catIds=person_cat_ids)
    return coco.loadAnns(annotation_ids)

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
#  bounding_box_count: int - number of bounding boxes per cell; known as B in YOLO paper
#    NOTE: Dead paramater for now - only works with one
#  cell_width: int - the width in pixels of a cell in the image.
#  cell_height: int - the height in pixels of a cell in the image.
#  img_id: id - id for the image
# Produces:
#  output: tensor[double] - A tensor of training data
# Preconditions:
#  coco is initialized with valid data
#  cell_width < width of image
#  cell_height < height of image
#  bounding_box_count >= 1
def gen_training_tensors(coco, bounding_box_count, cell_width, cell_height, img_id):
    # Force bounding_box_count to 1 - See NOTE in above documentation
    bounding_box_count = 1

    # These should probably get moved to the parameters
    DEFAULT_LOCATION = 0
    NO_OBJECT_WEIGHT = 0
    HAS_OBJECT_WEIGHT = 1

    # Position of the various training parameters along the last dimension
    # of the output data from the neural network
    POS_BOX_CENTER_X = 0
    POS_BOX_CENTER_Y = 1
    POS_BOX_WIDTH = 2
    POS_BOX_HEIGHT = 3
    POS_OBJ_LIKELYHOOD = 4

    annotations = get_image_annotations(coco, img_id)

    cell_x_count = ceil(img.shape[0] / cell_width)
    cell_y_count = ceil(img.shape[1] / cell_height)
    # 5 parameters to each bounding box: Probability, X pos, Y pos, Width, Height
    training_data = np.full((cell_x_count, cell_y_count, bounding_box_count * 5), DEFAULT_LOCATION)
    # Set all object probabilities to NO_OBJECT_WEIGHT
    if DEFAULT_LOCATION != NO_OBJECT_WEIGHT:
        training_data[..., ..., 4: :5] = NO_OBJECT_WEIGHT

    for annotation in annotations:
        # Calculate the cell that the annotation should match
        bounding_box = annotation['bbox']
        corner_x = int(bounding_box[0])
        corner_y = int(bounding_box[1])
        width = int(bounding_box[2])
        height = int(bounding_box[3])

        # Find the center of the box in terms of the whole image
        # These values are purposely floats to keep as much information as
        #  possible about the center of the image
        center_x = corner_x + width / 2
        center_y = corner_y + height / 2

        # Calculate the cell the bounding box is centered in
        cell_x_pos = floor(center_x / cell_width)
        cell_y_pos = floor(center_y / cell_height)

        # Find the center of the box relative to the corner of the cell:
        center_rel_x = center_x - (cell_x_pos * cell_width)
        center_rel_y = center_y - (cell_y_pos * cell_height)

        # ...And put it in terms of the cell size
        center_rel_x = center_rel_x / cell_width
        center_rel_y = center_rel_y / cell_height

        # Find the size of the bounding box relative to the cell
        rel_width = width / cell_width
        rel_height = height / cell_height

        # TODO: Move to handling more than one bounding box
        if training_data[cell_x_pos, cell_y_pos, 4] != NO_OBJECT_WEIGHT:
            logging.warn("Image %d has multiple bounding boxes in cell (%d,%d)" % (
                img_id,
                cell_x_pos,
                cell_y_pos
            ))

        # Set values for the training data
        training_data[cell_x_pos, cell_y_pos, POS_BOX_CENTER_X] = center_rel_x
        training_data[cell_x_pos, cell_y_pos, POS_BOX_CENTER_Y] = center_rel_y
        training_data[cell_x_pos, cell_y_pos, POS_BOX_WIDTH] = rel_width
        training_data[cell_x_pos, cell_y_pos, POS_BOX_HEIGHT] = rel_height
        training_data[cell_x_pos, cell_y_pos, POS_OBJ_LIKELYHOOD] = HAS_OBJECT_WEIGHT

    return training_data



# Procedure:
#  image_gen_factory
# Purpose:
#  To create a generator that yields numpy arrays corresponding to image ids
# Parameters:
#  image_ids: [int] - list of image ids to return
#  buffer_size: int = 0 - buffer images ahead of time
# Produces:
#  output: generator(np.array(float32)) - generator of numpy image arrays
# Preconditions:
#  image_ids refer to actual images on disk as defined in file_management.py
# Postconditions:
#  image_id is read in with the thing
def image_gen_factory(image_ids, buffer_size=0):
    for id in image_ids:
        yield file_management.get_image(id)

# Returns a generator of tuples: (img, training tensor)
# Normalize the image matrix (set all the value in the range
# [0, 1]) We should discus whether we want to standardize (z-score) our data or
# not.
# Returns a tuple of generators
def get_training_data_generators(cell_rows, cell_columns, bounding_box_count, image_width, image_height):
    ## Load in image data
    img_array = file_management.get_image(img['id'])

    # Compress to 0 to 1
    img_array = np.divide(img_array, 256, dtype=np.float32)
