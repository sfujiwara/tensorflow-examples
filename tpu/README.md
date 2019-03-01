
```
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" "https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig"
```

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