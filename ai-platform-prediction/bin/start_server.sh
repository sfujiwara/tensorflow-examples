IMAGE="mnist-tfserving"

docker run \
  --rm \
  -v "$(PWD)/outputs:/outputs" \
  ${IMAGE} \
  --port=8500 \
  --model_name="mnist" \
  --model_base_path="/models"

# docker run --rm -it -v "$(PWD)/outputs:/outputs" --entrypoint /bin/bash mnist-tfserving
