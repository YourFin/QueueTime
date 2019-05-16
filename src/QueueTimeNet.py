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
from annotations import cnn_y_to_absolute


# should output a 10*10*5 tensor
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
	# should result in 160*160*64

	# Conv. Layer 3x3x192
	model.add(Conv2D(192, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# should result in 80*80*192

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
	# should result in 40*40*512

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
	# should result in 20*20*1024

	# Conv. Layer * 2*2 + 2
	model.add(Conv2D(512, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(512, (1, 1), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), strides = 2, padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	# should result in 10*10*1024

	# Conv. Layer * 2
	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))

	model.add(Conv2D(1024, (3, 3), padding="same"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(LeakyReLU(alpha=0.1))
	# should result in 10*10*1024

	# first FC
	# model.add(Flatten())
	# model.add(Dense(4096))
	# model.add(Dense(classes))

	#z said to use conv layer instead of fc layer, blame her if this is wrong
	model.add(Dropout(0.5))
	model.add(Conv2D(5, (3, 3), padding="same"))
	model.add(Activation("relu"))
	# should result in 10*10*5
	# model.add(BatchNormalization())

	# softmax classifier
	# model.add(Activation("softmax"))

	# return the constructed network architecture
	return model

def QueueTime_loss(y_true, y_pred): # should be a BS * CELL_ROW * CELL_COL * 5 tensor
	# each one of them should now be batch*10*10*5
	print("[INFO] ytrue", y_true)
	print("[INFO] ypred", y_pred)

	y_true = K.reshape(y_true, [-1, 10, 10, 5])
	y_pred = K.reshape(y_pred, [-1, 10, 10, 5])

	print("[INFO] ytrue", y_true)
	print("[INFO] ypred", y_pred)

	coord = 3
	noobj = 0.1

	indicator = y_true[...,0]
	print("[INFO] indicator", indicator)
	x_loss = K.square(y_true[...,1] - y_pred[...,1]) 
	# print("[INFO] x loss", x_loss.eval())
	y_loss = K.square(y_true[...,2] - y_pred[...,2])
	# print("[INFO] y loss", y_loss.eval())
	xy_loss = coord * indicator * (y_loss+x_loss)
	print("[INFO] xy_loss ? 10 10 [[[1]]]", xy_loss)


	w_loss = K.square(K.sqrt(y_true[...,3]) - K.sqrt(y_pred[...,3])) #hard code now
	h_loss = K.square(K.sqrt(y_true[...,4]) - K.sqrt(y_pred[...,4])) #hard code now
	wh_loss = coord * indicator*(w_loss+h_loss)

       
	pred_box_xy = y_pred[..., 1:3]
	print("[INFO] pred_box_xy ?, 10, 10, 2", pred_box_xy)
	pred_box_wh = y_pred[..., 3:5]
	true_box_xy = y_true[..., 1:3] # relative position to the containing cell
	true_box_wh = y_true[..., 3:5] # number of cells accross, horizontally and vertically
	print("[INFO] pred_box_wh ? 10 10 2", pred_box_wh)
	
	### adjust confidence
	true_wh_half = true_box_wh / 2.
	true_mins    = true_box_xy - true_wh_half
	true_maxes   = true_box_xy + true_wh_half
	print("[INFO] true_maxes ? 10 10 2", true_maxes)

	pred_wh_half = pred_box_wh / 2.
	pred_mins    = pred_box_xy - pred_wh_half
	pred_maxes   = pred_box_xy + pred_wh_half       

	intersect_mins  = K.maximum(pred_mins,  true_mins)
	intersect_maxes = K.minimum(pred_maxes, true_maxes)
	intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
	intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
	print("[INFO] intersect_areas ? 10 10", intersect_areas)
	true_areas = true_box_wh[..., 0] * true_box_wh[..., 1] #may equal to 0
	pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1] #may equal to 0

	union_areas = pred_areas + true_areas - intersect_areas
	iou_scores  = tf.truediv(intersect_areas, union_areas + 1e-10)
	
	# true_box_conf = iou_scores * y_true[..., 0]
	# print("[INFO] true_box_conf ? 10 10", true_box_conf) #expect ?*10*10 here
	

	pr_loss_pos = indicator * K.square(iou_scores * y_true[..., 0] - y_pred[...,0])
	pr_loss_neg = noobj*(1-indicator) * K.square(iou_scores * y_true[..., 0] - y_pred[...,0])
	print("[INFO] pr_loss_neg ? 10 10", pr_loss_neg) #expect ?*10*10 here

	# m = K.int_shape(y_true)
	# print("[INFO] y_true is ", y_true, ",m is ", m, "xy_loss is", xy_loss[0])


	loss = (xy_loss+wh_loss+pr_loss_neg+pr_loss_pos)/16.0 #hard code BS now
	# fake_loss = pr_loss_neg

	real_loss = K.sum(K.sum(K.sum(loss,0), 0), 0, True)
	print("[INFO] real_loss", real_loss)
	return real_loss
	

# get rid of the duplicates using non-max suppression. also filter out the boxes
# with very low score (And calculate the iou between pred and ground truth???)
def QueueTime_post_process(y_pred, max_boxes_count = 15, iou_threshold = 0.7, score_threshold = 0.5): # y_pred should be a 10*10*5 tensor
	flatten_absolute_list = cnn_y_to_absolute(64, 64, y_pred) #hard code now!
	scores = np.empty((100, 1)) #hard code now!
	absolute_boxes = np.empty((100, 4)) #hard code now!
	counter = 0
	for entry in flatten_absolute_list:
		scores[counter, 0] = entry['likelyhood']
		# tensorflow needs [y1, x1, y2, x2]
		absolute_boxes[counter, 1] = entry['bbox'][0] #x1
		absolute_boxes[counter, 0] = entry['bbox'][1] #y1
		absolute_boxes[counter, 3] = entry['bbox'][0] + entry['bbox'][2] #x2 = x1 + w
		absolute_boxes[counter, 2] = entry['bbox'][1] + entry['bbox'][3] #y2 = y1 + h
	
	#pass the threshold for score
	scores_tf = tf.convert_to_tensor(scores)
	absolute_boxes_tf = tf.convert_to_tensor(absolute_boxes)
	prediction_mask = scores_tf >= score_threshold
	scores_tf = tf.boolean_mask(scores_tf, prediction_mask)
	absolute_boxes_tf = tf.boolean_mask(absolute_boxes_tf, prediction_mask)

	#after gathering
	#nms_index should be 1*15
	#scores_tf should be 15*1
	#absolute_boxes_tf should be 15*4
	nms_index = tf.image.non_max_suppression(
        absolute_boxes_tf, scores_tf, max_boxes_count, iou_threshold=iou_threshold)
	scores_tf = tf.gather(scores_tf, nms_index)
	absolute_boxes_tf = tf.gather(absolute_boxes_tf, nms_index)

	# convert two tfs back into numpy and score x1 y2 w h format
	post_pred = []
	scores_np = scores_tf.eval()
	absolute_boxes_np = absolute_boxes_tf.eval()
	counter = 0
	for i in range(scores_np.shape[0]):
		x1 = absolute_boxes_np[i][1]
		y1 = absolute_boxes_np[i][0]
		w = absolute_boxes_np[i][3] - absolute_boxes_np[i][1]
		h = absolute_boxes_np[i][2] - absolute_boxes_np[i][0]
		entry = {'score' : scores_np, 'bbox' : [x1, y1, w, h]}
		post_pred.append(entry)
	
	return post_pred




