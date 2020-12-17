#!/bin/sh

PROJECT_ID="sfujiwara"
REGION="us-central1"
# IMAGE="gcr.io/${PROJECT_ID}/mnist-tfserving"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/tfserving/mnist"
MODEL="mnist_tfserving"

gcloud beta artifacts repositories create tfserving \
  --project=${PROJECT_ID} \
  --repository-format=docker \
  --location=${REGION}

docker push ${IMAGE}

gcloud beta ai-platform models create ${MODEL} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --enable-logging \
  --enable-console-logging

gcloud beta ai-platform versions create v2 \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --model=${MODEL} \
  --image=${IMAGE}:latest \
  --machine-type=n1-standard-4 \
  --args="--rest_api_port=8501 --model_name=mnist --model_base_path=/models"

