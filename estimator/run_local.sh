#!/usr/bin/env bash

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path trainer \
  --distributed \
  --parameter-server-count 1 \
  --worker-count 1 \
  -- \
  --distribute_strategy="collective_all_reduce" \
  --max_steps=500 \
  --model_dir="outputs" \
  --num_gpus_per_worker=0 \
  --save_steps=250
