# TensorFlow Examples Using TPUs

## Run on Compute Engine and Cloud TPU

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
MODEL_DIR=${YOUR_CLOUD_STORAGE_PATH}
TPU_NAME=${YOUR_TPU_NAME}
```

```bash
python mnist_tpu.py \
  --use_tpu \
  --tpu_name=${TPU_NAME} \
  --model_dir=${MODEL_DIR} \
  --max_steps=10000 \
  --save_steps=1000
```

## Run on Cloud ML Engine

```
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
JOB_NAME=tpu_1
STAGING_BUCKET="gs://${PROJECT_ID}-mlengine"
REGION=us-central1
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
        --staging-bucket $STAGING_BUCKET \
        --runtime-version 1.12 \
        --scale-tier BASIC_TPU \
        --module-name mnist.mnist_tpu \
        --package-path mnist \
        --region $REGION
```