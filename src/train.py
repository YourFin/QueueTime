# This code is adapted from
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# # set the matplotlib backend so figures can be saved in the background
from preprocessing import get_training_data_generator
from QueueTimeNet import build, QueueTime_loss

# import the necessary packages
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

DATA_SIZE = 64115
EPOCHS = 5
INIT_LR = 1e-3   #0.001 learning_rate
BS = 32
IMAGE_DIMS = (640, 640, 3)
CELL_ROW = 3
CELL_COL = 20
BOUNDING_BOX_COUNT = 1
NUM_CLASSES = 1

# initialize the data and labels
# data = []
# labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)

# loop over the input images
# for imagePath in imagePaths:
# 	# load the image, pre-process it, and store it in the data list
# 	image = cv2.imread(imagePath)
# 	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
# 	image = img_to_array(image)
# 	data.append(image)

# 	# extract the class label from the image path and update the
# 	# labels list
# 	label = imagePath.split(os.path.sep)[-2]
# 	labels.append(label)

(data, labels) = get_training_data_generator(CELL_ROW,CELL_COL,BOUNDING_BOX_COUNT)

# scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=NUM_CLASSES)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=QueueTime_loss, optimizer=opt,
	metrics=["accuracy"]) # not sure about the metrics, decided later

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
# print("[INFO] serializing label binarizer...")
# f = open(args["labelbin"], "wb")
# f.write(pickle.dumps(lb))
# f.close()

# plot the training loss and accuracy
# check this later
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
