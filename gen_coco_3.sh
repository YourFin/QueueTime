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

if ! [ -d ./cocapi/PythonAPI ] ; then
    git submodule init
    git submodule update
fi

mkdir -p ./lib/
# Copy over the directory if it doesn't exist
cp -r --remove-destination ./cocoapi ./lib/

cd lib/cocoapi/PythonAPI || exit 1

# Convert it to python3
2to3 . -w >/dev/null
