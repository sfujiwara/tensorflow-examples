#!/usr/bin/env bash

PROJECT_ID=`gcloud config list project --format "value(core.project)"`
JOB_NAME="tf`date '+%Y%m%d%H%M%S'`"

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-mlengine-staging" \
  --region=us-central1 \
  --config=configs/gpu1_worker1.yaml \
  -- \
  --batch_size=64 \
  --learning_rate=0.01 \
  --max_steps=10000 \
  --model_dir="gs://${PROJECT_ID}-mlengine/${JOB_NAME}" \
  --save_steps=2500 \
  --tfds_dir="gs://${PROJECT_ID}-tfds" \
  --tfhub_dir="gs://${PROJECT_ID}-tfhub"
