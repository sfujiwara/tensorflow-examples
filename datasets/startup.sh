#!/usr/bin/env bash

pip3 install tensorflow-datasets

PROJECT_ID=`gcloud config list project --format "value(core.project)"`
DATASET=imagenet2012
TFDS_DIR=gs://${PROJECT_ID}-tfds

python3 -c "import tensorflow_datasets as tfds; tfds.load('${DATASET}', data_dir='${TFDS_DIR}')"
