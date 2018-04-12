import numpy as np
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.examples.tutorials.mnist import input_data


# Setting for TPU
USE_TPU = True
TPU_NAMES = ["demo-tpu"]
NUM_SHARDS = 8

# You should use Google Cloud Storage when using Cloud TPU
MODEL_DIR = "model"


def model_fn(features, labels, mode, params):
    # Build graph
    logits = tf.layers.dense(features, 10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    optim = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    if params["use_tpu"]:
        optim = tpu.CrossShardOptimizer(optim)
    train_op = optim.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
    # Create EstimatorSpec
    estimator_spec = tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
    )
    return estimator_spec


def train_input_fn(params):
    mnist = input_data.read_data_sets("data", one_hot=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (mnist.train.images.astype(np.float32), mnist.train.labels.astype(np.float32))
    )
    ds = ds.repeat()
    ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params["batch_size"]))
    images, labels = ds.make_one_shot_iterator().get_next()
    return images, labels


def eval_input_fn(params):
    mnist = input_data.read_data_sets("data", one_hot=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (mnist.train.images.astype(np.float32), mnist.train.labels.astype(np.float32))
    )
    ds = ds.repeat()
    ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params["batch_size"]))
    images, labels = ds.make_one_shot_iterator().get_next()
    return images, labels


def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    if USE_TPU:
        # tpu_names ==> tpu in TensorFlow v1.7
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=TPU_NAMES,
            zone="us-central1-c",
        )
        master = tpu_cluster_resolver.get_master()
    else:
        master = ""
    tf.logging.debug("Master: {}".format(master))

    # Create TPU config
    tpu_config = tpu.TPUConfig(num_shards=NUM_SHARDS)

    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=USE_TPU,
        train_batch_size=32,
        eval_batch_size=32,
        params={"use_tpu": USE_TPU},
        config=run_config,
    )

    estimator.train(
        input_fn=train_input_fn,
        max_steps=100000,
    )


if __name__ == "__main__":
    main()
