# Example of Estimator

## Run on Local

```bash
python -m trainer.task
```

## Run on Google Cloud ML Engine

### Setting

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_BUCKET=gs://${PROJECT_ID}-ml
```

```bash
gsutil mb gs://${PROJECT_ID}-tfds
```

```bash
JOB_NAME="tf`date '+%Y%m%d%H%M%S'`"

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=config.yaml
```
