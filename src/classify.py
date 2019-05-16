# This code is adapted from
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.models import load_model
import argparse
from file_management import get_image
from annotations import cnn_y_to_absolute, plot_annotations
from QueueTimeNet import QueueTime_loss, QueueTime_post_process
from keras.utils.generic_utils import get_custom_objects
from preprocessing import pad_image, PADDED_SIZE
from train import CELL_WIDTH, CELL_HEIGHT
from mAP_formatting import classified_write_anns_to_file
import numpy as np
import matplotlib as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
                help="input image id", type=int)
ap.add_argument("-f", "--ap-file", action='store_true',
                help='Write out classification to corresponding mAP file')
ap.add_argument("-p", "--no-plot", action='store_false',
                help='If this flag is passed, do not display the plots')
args = vars(ap.parse_args())

img_id = args["image"]

image = pad_image(get_image(img_id), PADDED_SIZE)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
# Load in the custom loss function
get_custom_objects().update({"QueueTime_loss": QueueTime_loss})

model = load_model(args["model"])

# classify the input image
print("[INFO] classifying image...")
y_pred = model.predict(image)[0]
# post_pred = QueueTime_post_process(y_pred)
post_pred = cnn_y_to_absolute(CELL_WIDTH, CELL_HEIGHT, y_pred)

# filter out all scores below threshold:
post_pred_filtered = list(filter(lambda ann: ann['score'] > 0.001, post_pred))

# Add color key:
scores = [ann['score'] for ann in post_pred_filtered]
normalize_score = lambda score: (score - min(scores)) / (max(scores) - min(scores) + 1e-10)
for ann in post_pred_filtered:
    ann['normalized_score'] = normalize_score(ann['score'])
    ann['color'] = plt.cm.jet(normalize_score(ann['score']))
print('Max score (dark red): ' + str(max(scores)))
print('Min score (dark blue): ' + str(min(scores)))

if not args['no_plot']:
    plot_annotations(img_id, post_pred_filtered)
if args['ap_file']:
    classified_write_anns_to_file(args['image'], post_pred_filtered)




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
