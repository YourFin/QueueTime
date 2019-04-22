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

4. Run `./docker.sh`

For a python shell with access to all of the libraries used, run:

```bash
./docker.sh python
```

And for a bash instance under docker, run:

```bash
./docker.sh shell
```
### Installing locally with pipenv (requires python 3.7) - required for gui use
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

5. Run `queuetime`:

```bash
pipenv run queuetime
```

Alternatively the virtual environment can be opened with `pipenv shell`, and then queuetime can be run with `./queuetime`
