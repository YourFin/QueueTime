#!/usr/bin/env python3
from queuefinding import abs_anns_to_heatmap, heatmap_bounding_box_sum
from playback_labels import playback_with_labels
import cv2

if __name__ == '__main__':
    import os
    import argparse
    import json
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=Path, help="Path to input video")
    ap.add_argument("-a", "--annotations", type=Path, help="Path to annotations file")
    ap.add_argument("-s", "--start_frame", type=int, help="Start frame number. defaults to 0",
                    default=0)
    ap.add_argument("-e", "--end_frame", type=int, help="End frame number. defaults to -1",
                    default=-1)
    ap.add_argument("-t", "--threshold", type=float, help="Threshold for displaying frames",
                    default=1)
    ap.add_argument("-d", "--std-dev", type=float, help="Std dev. for gaussian kernel",
                    default=1)
    ap.add_argument("-k", "--kernel-size", type=float, help="Gaussian Kernel size",
                    default=5)
    ap.add_argument("-m", "--display-heat-map", action='store_true', help="display the heatmap")
    # Annotations file should be a json file with the format:
    # [[{'bbox': [x,y,w,h], 'score': float}]]
    #   - outer list is by frame, inner list is for each annotation
    arguments = vars(ap.parse_args())

    if arguments['end_frame'] == -1:
        arguments['end_frame'] = None

    assert os.path.exists(arguments['annotations']), "Annotations file does not exist"
    assert os.path.exists(arguments['video']), "Video file does not exist"

    with open(arguments['annotations']) as json_file:
        annotations = json.load(json_file)

    vidstream = cv2.VideoCapture(str(arguments['video']))
    ret, frame = vidstream.read()
    if not ret:
        print("video does not exist")
        exit(1)
    (rows, cols, _) = frame.shape
    heatmap = abs_anns_to_heatmap(cols, rows,
                                  [ann for frame in annotations[arguments['start_frame']:arguments['end_frame']] for ann in frame])

    if arguments['display_heat_map']:
        print('hello')
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.imshow(heatmap) #, cmap=cm.jet)
        plt.show()

    def filt(_, ann):
        score = heatmap_bounding_box_sum(heatmap, ann['bbox'])
        #print(score)
        return score > arguments['threshold']

    playback_with_labels(str(arguments['video']), annotations,
                         start_frame=arguments['start_frame'], end_frame=arguments['end_frame'],
                         annotation_filter=filt)
