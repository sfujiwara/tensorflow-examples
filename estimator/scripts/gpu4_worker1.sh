#!/usr/bin/env bash

PROJECT_ID=`gcloud config list project --format "value(core.project)"`
JOB_NAME="tf`date '+%Y%m%d%H%M%S'`"

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-mlengine-staging" \
  --region=us-central1 \
  --config=configs/gpu4_worker1.yaml \
  -- \
  --batch_size=64 \
  --distribute_strategy="mirrored" \
  --max_steps=10000 \
  --model_dir="gs://${PROJECT_ID}-mlengine/${JOB_NAME}" \
  --num_gpus_per_worker=4 \
  --save_steps=2500 \
  --tfds_dir="gs://${PROJECT_ID}-tfds"
