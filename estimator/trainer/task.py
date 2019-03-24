import argparse
import json
import os
import tensorflow as tf
from . import model, pipeline, compat


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--distribute_strategy', default=None, type=str)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--num_gpus_per_worker', default=0, type=int)
parser.add_argument('--save_steps', type=int)
parser.add_argument('--tfds_dir', default=None, type=str)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
DISTRIBUTE_STRATEGY = args.distribute_strategy
MAX_STEPS = args.max_steps
MODEL_DIR = args.model_dir
NUM_GPUS_PER_WORKER = args.num_gpus_per_worker
SAVE_STEPS = args.save_steps
TFDS_DIR = args.tfds_dir


def main():

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info(tf.__version__)

    params = {
        'optimizer': tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    }

    # Set DistributeStrategy
    if DISTRIBUTE_STRATEGY == 'mirrored':
        compat.replace_master_with_chief()
        compat.replace_ps_with_evaluator()
        compat.delete_tf_config()
        # distribute = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
        distribute = tf.distribute.MirroredStrategy()
    elif DISTRIBUTE_STRATEGY == 'collective_all_reduce':
        compat.replace_master_with_chief()
        compat.replace_ps_with_evaluator()
        distribute = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
    elif DISTRIBUTE_STRATEGY == 'parameter_server':
        compat.replace_master_with_chief()
        distribute = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
    else:
        distribute = None

    # Show TF_CONFIG
    tf_conf = json.loads(os.environ.get("TF_CONFIG", "{}"))
    tf.logging.info("TF_CONFIG: {}".format(json.dumps(tf_conf, indent=2)))

    config = tf.estimator.RunConfig(
        save_summary_steps=SAVE_STEPS,
        save_checkpoints_steps=SAVE_STEPS,
        keep_checkpoint_max=10,
        train_distribute=distribute,
    )

    tf.logging.info('Cluster Spec: {}'.format(config.cluster_spec.as_dict()))

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=MODEL_DIR,
        config=config,
        params=params,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=pipeline.create_train_input_fn(tfds_dir=TFDS_DIR, batch_size=BATCH_SIZE),
        max_steps=MAX_STEPS,
        hooks=None,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=pipeline.create_eval_input_fn(tfds_dir=TFDS_DIR),
        steps=None,
        start_delay_secs=0,
        throttle_secs=5,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
