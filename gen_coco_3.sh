#!/bin/bash
if ! type 2to3 &>/dev/null ; then
    echo "2to3 not installed; you're going to need to run this in docker, or install it"
    exit 1
fi

# Move to script directory
function finish {
    popd >/dev/null
}
pushd "$(dirname "$0")" >/dev/null || exit 1
trap finish EXIT

# Copy over the directory if it doesn't exist
[ -d "./src/cocoapi" ] || cp -r cocoapi/PythonAPI src/cocoapi

function remove_common_coco_files {
    rm -rf cocoapi/common
    finish
}
trap remove_common_coco_files EXIT
cp -r cocoapi/common src/common || exit 1

cd src/cocoapi

# Convert it to python3
2to3 . -w >/dev/null
