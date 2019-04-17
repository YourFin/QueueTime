FROM python:3.7
LABEL maintainer="github:YourFin"

  # See: https://github.com/pypa/pipenv/issues/1586#issuecomment-413597222
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY Pipfile.lock /app/
  #RUN pipenv lock --clear
COPY Pipfile /app/
  # See: https://github.com/pypa/pipenv/issues/2924
  #RUN pipenv run pip install pip==18.0
RUN pipenv install --system

ENV PYTHONUNBUFFERED 1

COPY . /app

CMD ['./queuetime']
