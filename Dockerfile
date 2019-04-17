FROM python:3.7
LABEL maintainer="github:YourFin"

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY Pipfile.lock /app/
#RUN pipenv lock --clear
COPY Pipfile /app/
RUN pipenv install --system

ENV PYTHONUNBUFFERED 1

COPY . /app

CMD ['./queuetime']
