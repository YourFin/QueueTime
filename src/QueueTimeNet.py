# This code is adapted from
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


# import the necessary packages
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

def build(width, height, depth, classes):
	# initialize the model along with the input shape to be
	# "channels last" and the channels dimension itself
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# Conv. Layer 7x7x64-s-2
	model.add(Conv2D(64, (7, 7), strides = 2, padding="same",
		input_shape=inputShape))
	model.add(BatchNormalization(axis=chanDim)) #order matters?
	model.add(LeakyReLU(alpha=0.1)) #order matters?
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Conv. Layer 3x3x192
	model.add(Conv2D(192, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	# Conv. Layers *4
	model.add(Conv2D(128, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(256, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(256, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv. Layers  4*2 + 2 
	model.add(Conv2D(256, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(256, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(256, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(256, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Conv. Layer * 2*2 + 2
	# model.add(Conv2D(512, (1, 1), padding="same"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# model.add(Conv2D(1024, (3, 3), padding="same"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# model.add(Conv2D(512, (1, 1), padding="same"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# model.add(Conv2D(1024, (3, 3), padding="same"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# model.add(Conv2D(1024, (3, 3), padding="same"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# model.add(Conv2D(1024, (3, 3), strides = 2, padding="valid"))
	# model.add(BatchNormalization(axis=chanDim))
	# model.add(LeakyReLU(alpha=0.1))

	# Conv. Layer * 2
	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	# first FC
	# model.add(Flatten())
	# model.add(Dense(4096))
	# model.add(Dense(classes))

	#z said to use conv layer instead of fc layer, blame her if this is wrong
	model.add(Dropout(0.5))
	model.add(Conv2D(5, (3, 3), padding="same"))
	model.add(Activation("relu"))
	# model.add(BatchNormalization())

	# softmax classifier
	# model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

def QueueTime_loss(y_true, y_pred): # should be a CELL_ROW * CELL_COL * 5 tensor
	# each one of them should now be batch*10*10*5
	print("[INFO] ytrue", y_true)
	print("[INFO] ypred", y_pred)

	y_true = K.reshape(y_true, [-1, 20, 20, 5])
	y_pred = K.reshape(y_pred, [-1, 20, 20, 5])

	print("[INFO] ytrue", y_true)
	print("[INFO] ypred", y_pred)

	coord = 5
	noobj = 0.5
	
	loss = 0
	print("[INFO] loss init", loss)
	xy_loss = 0
	wh_loss = 0

	indicator = y_true[...,0]
	print("[INFO] indicator", indicator)
	x_loss = K.square(y_true[...,1] - y_pred[...,1])
	print("[INFO] x loss", x_loss)
	y_loss = K.square(y_true[...,2] - y_pred[...,2])
	print("[INFO] y loss", y_loss)
	xy_loss += coord * K.sum(indicator*(y_loss+x_loss))
	print("[INFO] xy_loss", xy_loss)
	print("[INFO] test", K.get_value(xy_loss))
	w_loss = K.square(K.sqrt(y_true[...,3]) - K.sqrt(y_pred[...,3]))
	h_loss = K.square(K.sqrt(y_true[...,4]) - K.sqrt(y_pred[...,4]))
	wh_loss += coord * np.sum(indicator*(w_loss+h_loss),None)

	pr_loss_neg = 0
	pr_loss_pos = 0
	pr_loss_pos += K.sum(indicator * K.square(indicator - y_pred[...,0]))
	pr_loss_neg += noobj*K.sum((1-indicator) * K.square(indicator - y_pred[...,0]))
	
	# K.shape(x_loss)

	# m = K.int_shape(y_true)
	# print("[INFO] y_true is ", y_true, ",m is ", m, "xy_loss is", xy_loss[0])


	loss += (xy_loss+wh_loss+pr_loss_neg+pr_loss_pos)/32
	print("[INFO] loss", loss)
	return loss
	
	
	
