#!/usr/bin/env python3
from queuefinding import abs_anns_to_heatmap, heatmap_bounding_box_sum
import cv2
from time import sleep
from matplotlib.cm import get_cmap

if __name__ == '__main__':
    import os
    import argparse
    import json
    from matplotlib import cm
    import matplotlib.pyplot as plt
    from pathlib import Path

    colormap = get_cmap('jet')

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
    ap.add_argument('-c', '--frame-count', type=int, help='count of frames to use per heatmap', default=10)
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

    if arguments['display_heat_map']:
        heatmap = abs_anns_to_heatmap(cols, rows,
                                    [ann for frame in annotations[arguments['start_frame']:arguments['start_frame'] + arguments['frame_count']] for ann in frame])
        print('hello')
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.imshow(heatmap) #, cmap=cm.jet)
        plt.show()

    def filt(_, ann):
        score = heatmap_bounding_box_sum(heatmap, ann['bbox'])
        #print(score)
        return score > arguments['threshold']

    # Copy pasted from playback_labels.py due to time constaints
    video_path = str(arguments['video'])
    start_frame = arguments['start_frame']
    end_frame = arguments['end_frame']
    frame_count = arguments['frame_count']
    frame_delay = 0
    # Thickness of rectangles in pixels
    RECT_THICKNESS = 2

    IN_LINE_COLOR = (0,255,0)  # Green
    NOT_IN_LINE_COLOR = (0, 0, 255)  # Red

    vidstream = cv2.VideoCapture(video_path)

    frame_index = -1
    frames = []
    while vidstream.isOpened():
        frame_index += 1

        if end_frame is not None and frame_index >= end_frame:
            break

        ret, frame = vidstream.read()
        frames.append(frame)
        if frame_index < start_frame + frame_count:
            continue
        if not ret:
            break

        try:
            heatmap = abs_anns_to_heatmap(cols, rows,
                                        [ann for frame in annotations[frame_index-frame_count+1:frame_index+1] for ann in frame])
        except IndexError:
            break

        cv2.imshow('Video', colormap(heatmap))

        sleep(frame_delay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vidstream.release()
