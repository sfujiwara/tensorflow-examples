source bin/load_image_config.sh

# Push Docker image to Google Container Registry
docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
