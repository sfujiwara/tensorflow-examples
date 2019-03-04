import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.examples.tutorials.mnist import input_data


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use_tpu", action="store_true")
parser.add_argument("--tpu_name", type=str, default="")
parser.add_argument("--model_dir", type=str)
parser.add_argument("--max_steps", type=int)
parser.add_argument("--save_steps", type=int)
args, unknown_args = parser.parse_known_args()

# Setting for TPU
USE_TPU = args.use_tpu
TPU_NAME = args.tpu_name
MAX_STEPS = args.max_steps
SAVE_STEPS = args.save_steps
MODEL_DIR = args.model_dir


# Note:
# The key `batch_size` and `use_tpu` are automatically added to `params` by `TPUEstimator`.
def model_fn(features, labels, mode, params):

    # Build graph
    logits = tf.layers.dense(features, 10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    optim = tf.train.GradientDescentOptimizer(learning_rate=1e-2)

    # NOTE:
    # When using TPUs, you have to use CrossShardOptimizer which aggregate gradients with all reduce.
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


# Note:
# `input_fn` for `TPUEstimator` must include a `params` argument.
# The key `batch_size` is automatically added to `params` by `TPUEstimator`.
def train_input_fn(params):

    mnist = input_data.read_data_sets("data", one_hot=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (mnist.train.images.astype(np.float32), mnist.train.labels.astype(np.float32))
    )
    ds = ds.repeat()

    # Note:
    # `TPUEstimator` compile the subgraph and send it to TPUs.
    # Then, it can not treat dynamic graph and you must fix batch size.
    ds = ds.batch(params["batch_size"], drop_remainder=True)
    images, labels = ds.make_one_shot_iterator().get_next()
    return images, labels


def eval_input_fn(params):

    mnist = input_data.read_data_sets("data", one_hot=True)
    ds = tf.data.Dataset.from_tensor_slices(
        (mnist.train.images.astype(np.float32), mnist.train.labels.astype(np.float32))
    )
    ds = ds.repeat()
    ds = ds.batch(params["batch_size"], drop_remainder=True)
    images, labels = ds.make_one_shot_iterator().get_next()

    return images, labels


def main():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    if USE_TPU:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[TPU_NAME])
        master = tpu_cluster_resolver.get_master()
    else:
        master = ""
    tf.logging.debug("Master: {}".format(master))

    # Create TPU config
    # NOTE:
    # `iterations_per_loop` is interval to return from TPUs to host CPU
    # It is recommended to be set as number of global steps for next checkpoint.
    tpu_config = tpu.TPUConfig(iterations_per_loop=SAVE_STEPS)

    # Create Session config
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )

    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=MODEL_DIR,
        session_config=session_config,
        tpu_config=tpu_config,
        save_summary_steps=SAVE_STEPS,
        save_checkpoints_steps=SAVE_STEPS,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=USE_TPU,
        train_batch_size=32,
        eval_batch_size=32,
        config=run_config,
    )

    estimator.train(
        input_fn=train_input_fn,
        max_steps=MAX_STEPS,
    )


if __name__ == "__main__":
    main()
