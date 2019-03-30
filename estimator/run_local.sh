#!/usr/bin/env bash

PROJECT_ID=`gcloud config list project --format "value(core.project)"`

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path trainer \
  --distributed \
  --parameter-server-count 1 \
  --worker-count 3 \
  -- \
  --batch_size=8 \
  --learning_rate=0.01 \
  --max_steps=10 \
  --model_dir="outputs" \
  --save_steps=5 \
  --eval_steps=1 \
  --distribute_strategy="collective_all_reduce" \
  --num_gpus_per_worker=0 \
  --tfds_dir="gs://${PROJECT_ID}-tfds"
