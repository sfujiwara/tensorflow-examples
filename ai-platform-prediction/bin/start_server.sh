PROJECT_ID="sfujiwara"
REGION="us-central1"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/tfserving/mnist"

docker run \
  --rm \
  -p 8501:8501 \
  -v "$(PWD)/outputs:/outputs" \
  ${IMAGE} \
  --rest_api_port=8501 \
  --model_name="mnist" \
  --model_base_path="/models"

# docker run --rm -it -v "$(PWD)/outputs:/outputs" --entrypoint /bin/bash mnist-tfserving
