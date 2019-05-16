# QueueTime
Finds the length of lines of people in images - Grinnell CSC 262 Computer Vision Final Project

## Setup
### Installing locally with pipenv (requires python 3.7) - required for easy gui use
1. Install `pipenv`:
```bash
pip install --user pipenv
```

2. Get the COCO api ready:

```bash
git submodule init
git submodule update
./gen_coco_3.sh
```

3. Install dependencies:

```bash
pipenv install
```

4. Download the coco dataset:

```bash
pipenv run ./COCO_download.sh
```

5. Run pull up the virtualenv:

```bash
pipenv shell
```

### Installing with Docker (experimental)
1. Install [Docker](https://www.docker.com/get-started)
2. Grab the coco api

```bash
git submodule init
git submodule update
```

3. Run `./docker.sh download` to download the coco dataset

4. Run `./docker.sh shell` for a bash instance in the right virtualenv

## Usage:
Most executable python files have a help option available with `python3 $file --help`.
This will list the available arguments and what they do.
### Neural network portion:
#### Downloading images from coco:
```bash
cd src
python3 download.py [num_imgs]
```
#### Training:
```bash
cd src
python3 train.py -m $model_file -o 6000 -i 2000 -e 40 -b 10 -l 0.0005 -r True
```

#### Classifying:
`IMAGE_ID` should be one of the downloaded images:
```bash
cd src
python3 classify.py -m $model_file IMAGE_ID
```

### mAP scoring:
#### Generate mAP ground truth data:
Note: This file does not have help
```bash
cd src
python3 mAP_formatting.py IMAGE_ID IMAGE_ID IMAGE_ID IMAGE_ID ...
```
#### Generate model data:
Should be the same set of image ids as above
```bash
cd src
python3 classify.py -m $file -f -p IMAGE_ID IMAGE_ID IMAGE_ID IMAGE_ID ...
```
#### mAP scoring test:
```bash
cd mAP
python3 main.py
```
### Queue Classification
Note: the `queue-classification` folder has a different set of dependencies than the
rest of the project. As such, if you are in the larger python virtualenv when you enter
the folder, you should run the `exit` command before `cd`ing in, and then run `pipenv shell`
in the `queue-classification` folder.
#### Generate labels with pre-trained mask-rcnn
```bash
cd queue-classification
pipenv install
pipenv shell
python3 gen_labels.py -v $video_file -o "$video-file".json
```
#### Display video
```bash
cd queue-classification
python3 queue_classification.py -v $video-file -a "$video-file".json
```
