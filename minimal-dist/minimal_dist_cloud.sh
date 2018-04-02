#!/usr/bin/env bash

JOB_NAME="minimal_dist`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_PATH=gs://${PROJECT_ID}-mlengine/minimal-dist/${JOB_NAME}

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket=gs://${PROJECT_ID}-mlengine \
  --region=us-central1 \
  --config=config.yaml
