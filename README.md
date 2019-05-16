# QueueTime
Finds the length of lines of people in images - Grinnell CSC 262 Computer Vision Final Project

## Setup
### Installing with Docker (easier)
1. Install [Docker](https://www.docker.com/get-started)
2. Grab the coco api

```bash
git submodule init
git submodule update
```

3. Run `./docker.sh download` to download the coco dataset

4. Run `./docker.sh shell` for a bash instance in the right virtualenv

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

## File structure:
### Top level
