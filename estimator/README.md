# Example of Estimator

## Run on Local

```bash
python -m trainer.task
```

## Run on Google Cloud ML Engine

```bash
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
```

### Create Google Cloud Storage Buckets

```bash
# Create bucket for ML Engine
gsutil mb -c regional -l us-central1 gs://${PROJECT_ID}-mlengine
gsutil mb -c regional -l us-central1 gs://${PROJECT_ID}-mlengine-staging
# Create bucket for TensorFlow Datasets
gsutil mb -c regional -l us-central1 gs://${PROJECT_ID}-tfds
```

### Download Dataset to Cloud Storage

Unfortunately, [tqdm](https://github.com/tqdm/tqdm) occurs an error on Cloud ML Engine and [tensorflow-datasets](https://github.com/tensorflow/datasets) use it when automatically downloading dataset.
Therefore, we need to download datasets to Cloud Storage in advance.

I created an [issue](https://github.com/tensorflow/datasets/issues/310) and requested the option to turn tqdm off in tensorflow-datasets.

```bash
python download_data.py --tfds_dir gs://${PROJECT_ID}-tfds
```

### Run Training

```bash
bash run_cloud.sh
```

### Tips for Distribute Strategy and Cloud ML Engine

