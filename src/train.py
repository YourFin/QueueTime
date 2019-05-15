# This code is adapted from
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

# DATA_SIZE = 500 #out of 64115
# EPOCHS = 20
INIT_LR = 1e-3   #learning_rate
BS = 16
IMAGE_DIMS = (640, 640, 3)
CELL_ROW = 10
CELL_COL = 10
CELL_WIDTH = 64
CELL_HEIGHT = 64
BOUNDING_BOX_COUNT = 1
NUM_CLASSES = 1

if __name__ == '__main__':
    # USAGE
    # python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

    # # set the matplotlib backend so figures can be saved in the background
    # import the necessary packages
    # import matplotlib
    from keras.utils.generic_utils import get_custom_objects
    from keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam
    from keras.preprocessing.image import img_to_array
    # from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    # from pyimagesearch.smallervggnet import SmallerVGGNet
    import matplotlib.pyplot as plt
    # from imutils import paths
    import numpy as np
    import argparse
    import random
    # import pickle
    import cv2
    import os

    from pycocotools.coco import COCO
    from file_management import ANNOTATION_FILE, get_downloaded_ids
    from annotations import get_image_annotations, plot_annotations
    from preprocessing import all_imgs_numpy, all_ground_truth_numpy, training_data_generator
    from QueueTimeNet import build, QueueTime_loss

    # use custom loss
    # get_custom_objects.update({"QueueTime_loss": QueueTime_loss})
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    # ap.add_argument("-l", "--labelbin", required=True,
    # 	help="path to output label binarizer")
    ap.add_argument("-o", "--image_offset", type=int, default=3000)
    ap.add_argument("-i", "--image_count", type=int, default=1000)
    ap.add_argument("-e", "--epoch", type=int, default=40)
    ap.add_argument("-b", "--batch_size", type=int, default=10)
    ap.add_argument("-l", "--learning_rate", type=float, default=0.001)
    ap.add_argument("-r", "--reload", type=bool, default=False)
    # ap.add_argument("-p", "--plot", type=str, default="plot.png",
    #                 help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())

    EPOCHS = args["epoch"]
    BS = args["batch_size"]
    INIT_LR = args["learning_rate"]
    reload_bool = args["reload"]
    model_name = args["model"]
    model_name_for_save = model_name
    if (reload_bool == True):
        print("[INFO] reloading model")
        model_name_for_save = "new_trained" + model_name_for_save
    else:
        print("[INFO] training a new model")


    print("[INFO] loading images...")

    coco = COCO(ANNOTATION_FILE)
    opt = Adam(lr=INIT_LR, decay= 0) #INIT_LR / EPOCHS)

    if (reload_bool == False): 
        # print("[INFO] the true size", trainX.shape)

        # construct the image generator for data augmentation
        # aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        #                          horizontal_flip=True, fill_mode="nearest")

        # initialize the model
        print("[INFO] compiling model...")
        model = build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                    depth=IMAGE_DIMS[2], classes=NUM_CLASSES)
        model.compile(loss=QueueTime_loss, optimizer=opt, metrics=["accuracy"]) # not sure about the metrics, decided later
    else: 
        #Load partly trained model
        model = load_model(args["model"], custom_objects={ 'loss': QueueTime_loss })

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(
        training_data_generator(coco, args["image_offset"], args["image_count"], 1, CELL_WIDTH, CELL_HEIGHT, BS),
        # aug.flow(trainX, trainY, batch_size=BS),
        validation_data=training_data_generator(coco, args["image_offset"], args["image_count"] // 20, 1, CELL_WIDTH, CELL_HEIGHT, BS),
        validation_steps = args["image_count"] // (20 * BS), 
        steps_per_epoch=args["image_count"] // BS,
        epochs=args["epoch"], verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(model_name_for_save)

    # save the label binarizer to disk
    # print("[INFO] serializing label binarizer...")
    # f = open(args["labelbin"], "wb")
    # f.write(pickle.dumps(lb))
    # f.close()

    # plot the training loss and accuracy
    # check this later
    # plt.style.use("ggplot")
    # plt.figure()
    # N = EPOCHS
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="upper left")
    # plt.savefig(args["plot"])
