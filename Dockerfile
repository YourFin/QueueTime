FROM python:3.7
LABEL maintainer="github:YourFin"

RUN pip install --upgrade pip
RUN pip install pipenv 'setuptools>=18.0' cython

WORKDIR /app

# COCO build requirments
RUN pip install 'setuptools>=18.0' 'cython>=0.27.3' 'matplotlib>=2.1.0'
COPY gen_coco_3.sh /app/
COPY ./cocoapi /app/cocoapi/
RUN mkdir -p /app/src/
RUN ./gen_coco_3.sh

COPY Pipfile.lock /app/
#RUN pipenv lock --clear
COPY Pipfile /app/
RUN pipenv install --system

ENV PYTHONUNBUFFERED 1

COPY . /app

CMD ['./queuetime']
