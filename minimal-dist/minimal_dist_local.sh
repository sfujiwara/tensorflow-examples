#!/usr/bin/env bash

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path trainer \
  --distributed \
  --parameter-server-count 4 \
  --worker-count 8