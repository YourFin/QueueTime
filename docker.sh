#!/bin/bash

# Move to script directory
function finish {
    popd >/dev/null
}
pushd "$(dirname "$0")" >/dev/null || exit 1
trap finish EXIT

mkdir -p .version_test

# Copy the files over if they don't exist yet
if [ ! -f '.version_test/Pipfile' ]; then
    cp -n Pipfile .version_test/Pipfile >/dev/null
fi
if [ ! -f '.version_test/Pipfile.lock' ]; then
    cp -n Pipfile.lock .version_test/Pipfile.lock >/dev/null
fi

# If the files don't match the cache
if ! (cmp Pipfile .version_test/Pipfile &>/dev/null && cmp Pipfile .version_test/Pipfile &>/dev/null) ; then
    docker-compose build queuetime &&
        cp -f Pipfile .version_test/Pipfile &&
        cp -f Pipfile.lock .version_test/Pipfile.lock || exit 1
fi

if [ "$1" == "shell" ]; then
    docker-compose run queuetime bash
    if [ "$1" == "python" ]; then
        docker-compose run queuetime bash
else
    docker-compose up
fi
