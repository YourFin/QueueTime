#!/usr/bin/env python3
import json
import numpy as np
import cv2
from time import sleep

from os.path import dirname
import os

QUEUETIME_DIR = dirname(dirname(os.path.abspath(__file__)))

def playback_with_labels(video_path, annotations,
                         start_frame=0, end_frame=None,
                         annotation_filter=lambda frame_num, annotation: True,
                         frame_delay=0):
    """
    Play back the video at $video_path with $annotations rectangles on top

    $start_frame and $end_frame can be used to play back a subsection of the video
    $start_frame is inclusive and $end_frame is exclusive
    Defaults play back the entire video

    $annotation_filter is a function that is used to determine whether or not a
     bounding rectangle is "in line". True gets the box drawn in
     $IN_LINE_COLOR, while False get the box drawn in $NOT_IN_LINE_COLOR
     it is passed the index of the frame in the video, along with the
     annotation that is being drawn
     $annotation_filter defaults to a constant True function, i.e. all
      annotations shown

    To provide a way to slow down video playback, the video pauses for
    $frame_delay seconds between each frame.
    """

    # Thickness of rectangles in pixels
    RECT_THICKNESS = 2

    IN_LINE_COLOR = (0,255,0)  # Green
    NOT_IN_LINE_COLOR = (0, 0, 255)  # Red

    vidstream = cv2.VideoCapture(video_path)

    frame_index = -1
    while vidstream.isOpened():
        frame_index += 1

        if end_frame is not None and frame_index >= end_frame:
            break
        #frame_annotations = annotations[frame_index]

        ret, frame = vidstream.read()
        if frame_index < start_frame:
            continue
        if not ret:
            break

        try:
            for ann in annotations[frame_index]:
                if annotation_filter(frame_index, ann):
                    cur_color = IN_LINE_COLOR
                else:
                    cur_color = NOT_IN_LINE_COLOR

                bbox = ann['bbox']
                upper_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                cv2.rectangle(frame, upper_left, bottom_right, cur_color, RECT_THICKNESS)
        except IndexError:
            pass

        cv2.imshow('Video', frame)

        sleep(frame_delay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vidstream.release()

if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=Path, help="Path to input video")
    ap.add_argument("-a", "--annotations", type=Path, help="Path to annotations file")
    ap.add_argument("-s", "--start_frame", type=int, help="Start frame number. defaults to 0",
                    default=0)
    ap.add_argument("-e", "--end_frame", type=int, help="End frame number. defaults to -1",
                    default=-1)
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

    playback_with_labels(str(arguments['video']), annotations,
                         start_frame=arguments['start_frame'], end_frame=arguments['end_frame'])
