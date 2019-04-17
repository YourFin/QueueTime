#!/bin/bash

# Move to script directory
function finish {
    popd >/dev/null
}
pushd "$(dirname "$0")" >/dev/null || exit 1
trap finish EXIT

type curl &>/dev/null || (echo "curl not installed, please install it to continue or use docker" && exit 1)
type unzip &>/dev/null || (echo "unzip not installed, please install it to continue or use docker" && exit 1)

scratch=$(mktemp -d -t tmp.XXXXXXXXXX) || (echo "Error: could not creat temp directory" && exit 1)
function clean_scratch {
    rm -r $scratch
    finish
}

echo 'Downloading annotiations...'
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip > "$scratch/annotations.zip"
unzip "$scratch/annotations.zip" -d ./data/coco/
