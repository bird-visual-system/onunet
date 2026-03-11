#FROM jupyter/tensorflow-notebook:latest
FROM tensorflow/tensorflow:2.13.0-gpu

USER root

RUN apt-get update && apt-get install -y \
    python3-distutils \
    libproj-dev \
    gdal-bin

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy project:
WORKDIR /usr/src/onunet
COPY . /usr/src/onunet

CMD ["python", "main.py"]