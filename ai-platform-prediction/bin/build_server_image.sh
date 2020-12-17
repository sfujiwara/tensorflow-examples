PROJECT_ID="sfujiwara"
REGION="us-central1"
# IMAGE="gcr.io/${PROJECT_ID}/mnist-tfserving"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/tfserving/mnist"

docker build -t ${IMAGE} .
