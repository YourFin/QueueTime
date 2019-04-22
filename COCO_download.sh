#!/bin/bash

# Move to script directory
function finish {
    popd >/dev/null
}
pushd "$(dirname "$0")" >/dev/null || exit 1
trap finish EXIT

which curl &>/dev/null || which wget &>/dev/null || (echo "curl not installed, please install it to continue or use docker" && exit 1)
which unzip &>/dev/null || (echo "unzip not installed, please install it to continue or use docker" && exit 1)

scratch=$(mktemp -d -t tmp.XXXXXXXXXX) || (echo "Error: could not create temp directory" && exit 1)
function clean_scratch {
    rm -r $scratch
    finish
}

echo 'Downloading annotations...'
if which curl &>/dev/null ; then
    curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip > "$scratch/annotations.zip"
elif which wget &>/dev/null ; then
    wget --no-config http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$scratch/annotations.zip"
else
    echo "curl AND wget not installed, please install one to continue or use docker"
    exit 1
fi
unzip "$scratch/annotations.zip" -d ./data/coco/

python ./src/download.py
