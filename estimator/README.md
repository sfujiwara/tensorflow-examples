# Examples of Estimator

This is an example of Estimator and DistributeStrategy.

## Arguments

### Required Arguments

#### `--batch_size`

The number of samples which used for mini-batch training.
In case of synchronous distributed learning, each GPUs compute gradient with `batch_size`.
Therefore, the total number of samples is `batch_size * n_gpus`. 

#### `--learning_rate`

Learning rate which used for gradient descent optimizer.

#### `--max_steps`

Training will stop when `global_step` is greater than equal `max_step`.

#### `--model_dir`

Directory where checkpoint, summary, and SavedModel are saved to.

#### `--save_steps`

Checkpoint and summary are saved every `save_step`.

### Optional Arguments

#### `--distribute_strategy`

* mirrored
* collective_all_reduce
* parameter_server

#### `--num_gpus_per_worker`

#### `--tfds_dir`

#### `--tfhub_dir`

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
# Create bucket for TensorFlow Hub
gsutil mb -c regional -l us-central1 gs://${PROJECT_ID}-tfhub
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

## Tips for Distribute Strategy and Cloud ML Engine

Currently (2019-03) Cloud ML Engine can deal with three task types below:

* master
* worker
* ps

However, DistributeStrategy assumes four task types below:

* chief (not mater)
* worker
* ps (only `ParameterServerStrategy` use it)
* evaluator (optional)

Thus, we have to use two functions in [compat.py](trainer/compat.py):

* `replace_master_with_chief()`
  * Change master ==> chief in environment variable `TF_CONF`
* `replace_ps_with_evaluator()`
  * Change ps ==> evaluator in environment variable `TF_CONF`
  * We can use evaluator with `MirroredStrategy` and `CollectiveAllReduceStrategy` on Google Cloud ML Engine
 
 ## Benchmarks
 
 ### CollectiveAllReduceStrategy (num_gpus_per_worker: 1)
