#!/bin/bash
# Move to script directory
function finish {
    popd >/dev/null
}
pushd "$(dirname "$0")" >/dev/null || exit 1
trap finish EXIT

python classify_video_jit.py -v ../data/videos/1.mp4 -a 1.mp4.json -s 27 -e 170 -t 0.3 -m
