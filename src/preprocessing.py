import numpy as np

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
