#!/usr/bin/env bash

INSTANCE_NAME="tfds-downloader-`date '+%Y%m%d%H%M%S'`"

gcloud compute --project=sfujiwara instances create ${INSTANCE_NAME} \
  --zone=us-central1-b \
  --machine-type=n1-standard-1 \
  --subnet=default \
  --network-tier=PREMIUM \
  --maintenance-policy=MIGRATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server,https-server \
  --image=debian-9-tf-1-13-v20190329 \
  --image-project=ml-images \
  --boot-disk-size=1000GB \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=instance-1 \
  --metadata-from-file startup-script=startup.sh
