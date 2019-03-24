#!/usr/bin/env bash

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path trainer \
  --distributed \
  --parameter-server-count 1 \
  --worker-count 1 \
  -- \
  --batch_size=256 \
  --distribute_strategy="mirrored" \
  --max_steps=500 \
  --model_dir="outputs" \
  --num_gpus_per_worker=4 \
  --save_steps=250
