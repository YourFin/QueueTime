# QueueTime
Finds the length of lines of people in images - Grinnell CSC 262 Computer Vision Final Project

## Setup
### pipenv (requires python 3.7)
1. Install `pipenv`:

    pip install --user pipenv

2. Install dependencies:

    pipenv install

3. Build the coco api:

   git submodule init
   git submodule update
   ./gen_coco_3.sh

4. Run `queuetime`:

    pipenv run queuetime

Alternatively the virtual environment can be opened with `pipenv shell`, and then queuetime can be run with `./queuetime`

### Docker
1. Install [Docker](https://www.docker.com/get-started)
2. Grab the coco api

   git submodule init
   git submodule update

3. Run `docker.sh`
