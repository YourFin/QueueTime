# This code is adapted from
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.models import load_model
import numpy as np
import argparse
from file_management import get_image
from annotations import cnn_y_to_absolute, plot_annotations

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	            help="input image id", type=int)
args = vars(ap.parse_args())

img_id = args["image"]

image = get_image(img_id)
#image = np.expand_dims(image, axis=0) #?

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
# lb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)
anns = cnn_y_to_absolute(proba)
plot_annotations(img_id, anns)



# # we'll mark our prediction as "correct" of the input image filename
# # contains the predicted label text (obviously this makes the
# # assumption that you have named your testing image files this way)
# filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
# correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# # build the label and draw the label on the image
# label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
# output = imutils.resize(output, width=400)
# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
# 	0.7, (0, 255, 0), 2)

# # show the output image
# print("[INFO] {}".format(label))
# cv2.imshow("Output", output)
# cv2.waitKey(0)
