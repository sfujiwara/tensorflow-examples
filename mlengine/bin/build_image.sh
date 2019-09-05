source bin/load_image_config.sh

# Build Docker image
docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} .
