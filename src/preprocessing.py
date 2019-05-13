import logging
import numpy as np
from math import ceil, floor
import file_management
from annotations import get_image_annotations
from file_management import get_downloaded_ids, get_image

PADDED_SIZE = 640

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
def pad_image(img, size):
    (X_size, Y_size, _) = img.shape
    #X_size = img.shape[0]
    #Y_size = img.shape[1]

    return np.pad(img, ((0, size - X_size),(0, size - Y_size), (0, 0)), 'constant')  # Default to 0

# Procedure:
#  get_y_true
# Purpose:
#  To generate the ground truth corresponding to a given image.
# Parameters:
#  coco: COCO - a coco instance to pull annotation information from
#  bounding_box_count: int - number of bounding boxes per cell; known as B in YOLO paper
#    NOTE: Dead paramater for now - only works with one
#  cell_width_px: int - the width in pixels of a cell in the image.
#  cell_height_px: int - the height in pixels of a cell in the image.
#  img_id: id - id for the image
# Produces:
#  output: a numpy arr: cell_y_count * cell_x_count * 5
# Preconditions:
#  coco is initialized with valid data
#  cell_width_px < width of image
#  cell_height_px < height of image
#  bounding_box_count >= 1
def get_y_true(coco, bounding_box_count, cell_width_px, cell_height_px, img_id):
    # Force bounding_box_count to 1 - See NOTE in above documentation
    bounding_box_count = 1

    # These should probably get moved to the parameters
    DEFAULT_VALUE = 0
    NO_OBJECT_WEIGHT = 0
    HAS_OBJECT_WEIGHT = 1

    # Position of the various training parameters along the last dimension
    # of the output data from the neural network
    POS_OBJ_SCORE = 0
    POS_BOX_CENTER_X = 1
    POS_BOX_CENTER_Y = 2
    POS_BOX_WIDTH = 3
    POS_BOX_HEIGHT = 4

    annotations = get_image_annotations(coco, img_id)

    # img = coco.loadImgs([img_id])[0]

    # cell_x_count, how many cells are on horizontal direction, cell_y_count,
    # how many cells are on vertical direction
    cell_x_count = ceil(PADDED_SIZE / cell_width_px)
    cell_y_count = ceil(PADDED_SIZE / cell_height_px)
    # 5 parameters to each bounding box: Probability, X pos, Y pos, Width, Height
    y_true = np.full((cell_y_count, cell_x_count, bounding_box_count * 5), DEFAULT_VALUE)
    y_true = y_true.astype('float32')
    # Set all object probabilities to NO_OBJECT_WEIGHT
    if DEFAULT_VALUE != NO_OBJECT_WEIGHT:
        y_true[..., ..., POS_OBJ_SCORE: :5] = NO_OBJECT_WEIGHT

    for annotation in annotations:
        # Calculate the cell that the annotation should match
        bounding_box = annotation['bbox']

        # print("[DEBUG]", bounding_box)

        abs_ul_x = bounding_box[0]
        abs_ul_y = bounding_box[1]
        width = bounding_box[2]
        height = bounding_box[3]

        # Find the center of the box in terms of the whole image
        # These values are purposely floats to keep as much information as
        #  possible about the center of the image
        abs_center_x = abs_ul_x + width / 2
        abs_center_y = abs_ul_y + height / 2

        # Calculate the cell the bounding box is centered in
        cell_x_pos = floor(abs_center_x / cell_width_px)
        cell_y_pos = floor(abs_center_y / cell_height_px)

        # Find the center of the box relative to the corner of the cell:
        # ...And put it in terms of the cell size
        rel_center_x = (abs_center_x - (cell_x_pos * cell_width_px)) / cell_width_px
        rel_center_y = (abs_center_y - (cell_y_pos * cell_height_px)) / cell_height_px

        # Find the size of the bounding box relative to the cell
        rel_width = width / cell_width_px
        rel_height = height / cell_height_px

        # TODO: Move to handling more than one bounding box
        # if y_true[cell_y_pos, cell_x_pos, POS_OBJ_SCORE] != NO_OBJECT_WEIGHT:
        #     logging.warn("Image %d has multiple bounding boxes in cell (%d,%d)" % (
        #         img_id,
        #         cell_x_pos,
        #         cell_y_pos
        #     ))

        # Set values for the training data
        y_true[cell_y_pos, cell_x_pos, POS_BOX_CENTER_X] = rel_center_x
        y_true[cell_y_pos, cell_x_pos, POS_BOX_CENTER_Y] = rel_center_y
        y_true[cell_y_pos, cell_x_pos, POS_BOX_WIDTH] = rel_width
        y_true[cell_y_pos, cell_x_pos, POS_BOX_HEIGHT] = rel_height
        y_true[cell_y_pos, cell_x_pos, POS_OBJ_SCORE] = HAS_OBJECT_WEIGHT
        # print("[DEBUG]", y_true[cell_y_pos, cell_x_pos, :])
    return y_true

# Procedure:
#  y_true_generator
# Purpose:
#  To create a generator for ground truth data corresponding to image ids
# Parameters:
#  coco: COCO - a coco instance to pull annotation information from
#  bounding_box_count: int - number of bounding boxes per cell; known as B in YOLO paper
#    NOTE: Dead paramater for now - only works with one
#  cell_width_px: int - the width in pixels of a cell in the image.
#  cell_height_px: int - the height in pixels of a cell in the image.
#  image_ids: [int] - list of image ids to return
#  buffer_size: int = 0 - amount of data to buffer ahead of time
#    NOTE: Not implemented
#  save_data: bool = false - whether to save data to disk for recall
#    NOTE: Not implemented
#  load_data: bool = true - whether to check the disk for saved version of the training data
#    NOTE: Not implemented
# Produces:
#  output: generator(np.array(float32)) - generator of numpy arrays
# Preconditions:
#  COCO contains all given image ids
#  cell_width_px < min(width(imgs)) - should be fairly small
#  cell_hegiht < min(hegiht(imgs)) - should be fairly small
# Postconditions:
#  generator yields numpy arrays as defined get_y_true
def y_true_generator(
        coco,
        bounding_box_count,
        cell_width_px,
        cell_height_px,
        image_ids,
        buffer_size=0,
        save_data=False,
        load_data=True):
    for id in image_ids:
        yield get_y_true(
            coco,
            bounding_box_count,
            cell_width_px,
            cell_height_px,
            id)

# Procedure:
#  image_generator
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
#  All image ids exist
#  values in output are bounded between 0 and 1
def image_generator(image_ids, buffer_size=0):
    for id in image_ids:
        image = file_management.get_image(id)
        image = np.divide(image, 256, dtype=np.float32)
        try:
            image = pad_image(image, PADDED_SIZE)
        except Exception as e:
            print(id)
            raise e
        yield image

# Procedure:
#  all_imgs_numpy
# Purpose:
#  To return all images as a numpy array
# Parameters:
#  None
# Produces:
#  imgs: numpy(float) - first dim is image number, then row,column,color channel
# Precondations:
#  some images are download
# Postconditions:
#  out contains every image in data/coco
# Practica:
#  This is a bit of a hack, makes a lot of assumptions
def all_imgs_numpy(num_images):
    img_ids = get_downloaded_ids()
    img_ids = list(filter(is_not_greyscale, img_ids))[:num_images]
    imgs = np.empty((len(img_ids), 640, 640, 3), np.float)
    gen = image_generator(img_ids)
    for index in range(len(img_ids)):
        imgs[index, :, :, :] = next(gen)

    return imgs

# Procedure:
#  all_ground_truth_numpy
# Purpose:
#  Return all training data corresponding to downloaded images
# Parameters:
#  coco: COCO - a coco instance to pull annotation information from
#  bounding_box_count: int - number of bounding boxes per cell; known as B in YOLO paper
#    NOTE: Dead paramater for now - only works with one
#  cell_width_px: int - the width in pixels of a cell in the image.
#  cell_height_px: int - the height in pixels of a cell in the image.
# Produces:
#  output: numpy array of floats
# Preconditions:
#  no additional
# Postconditions:
#  all images defined on disk correspond to images defined here
def all_ground_truth_numpy(
        coco,
        num_images,
        bounding_box_count,
        cell_width_px,
        cell_height_px):
    #assert(cell_rows == ceil(PADDED_SIZE/cell_height_px))
    #assert(cell_columns == ceil(PADDED_SIZE/cell_width_px))
    img_ids = get_downloaded_ids()
    img_ids = list(filter(is_not_greyscale, img_ids))[:num_images]
    output = np.empty((len(img_ids), ceil(PADDED_SIZE/cell_height_px), ceil(PADDED_SIZE/cell_width_px), bounding_box_count * 5), np.float)
    gen = y_true_generator(coco, bounding_box_count, cell_width_px, cell_height_px, img_ids)
    for index in range(len(img_ids)):
        output[index, :, :, :] = next(gen)
    return output

# Returns a generator of tuple: (img, training tensor)
# Normalize the image matrix (set all the value in the range
# [0, 1]) We should discus whether we want to standardize (z-score) our data or
# not.
# Returns a tuple of generators
def training_data_generator(
        coco,
        start_index,
        num_images,
        bounding_box_count,
        cell_width_px,
        cell_height_px, 
        batch_size):
    img_ids = get_downloaded_ids()
    img_ids = list(filter(is_not_greyscale, img_ids))[:num_images]
    while 1:
        for img_id in img_ids:
            image_batch = np.empty((batch_size,640, 640, 3), np.float) #hard code
            y_true_batch = np.empty((batch_size,10,10,5), np.float) #hard code
            for i in range(batch_size): 
                image = file_management.get_image(img_id)
                image = np.divide(image, 256, dtype=np.float32)
                try:
                    image = pad_image(image, PADDED_SIZE)
                except Exception as e:
                    print(id)
                    raise e
                ground_truth = get_y_true(
                    coco,
                    bounding_box_count,
                    cell_width_px,
                    cell_height_px,
                    img_id
                )
                image_batch[i, :, :, :] = image
                y_true_batch[i, :, :, :] = ground_truth


            yield (image_batch, y_true_batch)





def is_not_greyscale(img_id):
    img = get_image(img_id)
    return img.ndim == 3
