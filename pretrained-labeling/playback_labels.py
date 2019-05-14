#!/usr/bin/env python3
import json
import numpy as np
import cv2

from os.path import dirname
import os

QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__)))

def playback_with_labels(video_path, annotations):
    "Play back the video at $video_path with $annotations rectangles on top"

    # Thickness of rectangles in pixels
    RECT_THICKNESS = 2

    vidstream = cv2.VideoCapture(video_path)

    frame_index = -1
    while vidstream.isOpened():
        frame_index += 1
        #frame_annotations = annotations[frame_index]

        ret, frame = vidstream.read()
        assert ret, "frame %d does not exist" % frame_index

        for ann in annotations[frame_index]:
            bbox = ann['bbox']
            upper_left = (bbox[0], bbox[1])
            bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), RECT_THICKNESS)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #plt.imshow(frame)
        #plt.show()
        #input("Press Enter to go to next frame...")

    vidstream.release()

if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=Path, help="Path to input video")
    ap.add_argument("-a", "--annotations", type=Path, help="Path to annotations file")
    # Annotations file should be a json file with the format:
    # [[{'bbox': [x,y,w,h], 'score': float}]]
    #   - outer list is by frame, inner list is for each annotation
    arguments = vars(ap.parse_args())

    assert os.path.exists(arguments['annotations']), "Annotations file does not exist"
    assert os.path.exists(arguments['video']), "Video file does not exist"

    with open(arguments['annotations']) as json_file:
        annotations = json.load(json_file)

    print(len(annotations))

    playback_with_labels(str(arguments['video']), annotations)
