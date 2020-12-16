source bin/load_image_config.sh

JOB_NAME="container_`date '+%Y%m%d%H%M%S'`"

MASTER_IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
WORKER_IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
PS_IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

gcloud ai-platform jobs submit training $JOB_NAME \
  --config config.yaml \
  --region us-west1 \
  --master-image-uri ${MASTER_IMAGE_URI} \
  --worker-image-uri ${WORKER_IMAGE_URI} \
  --parameter-server-image-uri ${PS_IMAGE_URI} \
  -- \
  --model-dir=gs://hoge \
  --epochs=10
