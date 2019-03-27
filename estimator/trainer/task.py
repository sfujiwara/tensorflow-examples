import argparse
import json
import os
import tensorflow as tf
from . import model, pipeline, compat


parser = argparse.ArgumentParser()

# Required
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--save_steps', type=int)

# Optional
parser.add_argument('--distribute_strategy', default=None, type=str)
parser.add_argument('--num_gpus_per_worker', default=None, type=int)
parser.add_argument('--tfds_dir', default=None, type=str)
parser.add_argument('--tfhub_dir', default=None, type=str)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
DISTRIBUTE_STRATEGY = args.distribute_strategy
LEARNING_RATE = args.learning_rate
MAX_STEPS = args.max_steps
MODEL_DIR = args.model_dir
NUM_GPUS_PER_WORKER = args.num_gpus_per_worker
SAVE_STEPS = args.save_steps
TFDS_DIR = args.tfds_dir
TFHUB_DIR = args.tfhub_dir


def select_distribute_strategy(distribute_strategy_name):

    # Set DistributeStrategy
    if distribute_strategy_name == 'mirrored':
        compat.replace_master_with_chief()
        compat.replace_ps_with_evaluator()
        compat.delete_tf_config()
        # distribute = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
        distribute = tf.distribute.MirroredStrategy()
    elif distribute_strategy_name == 'collective_all_reduce':
        compat.replace_master_with_chief()
        compat.replace_ps_with_evaluator()
        distribute = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
    elif distribute_strategy_name == 'parameter_server':
        compat.replace_master_with_chief()
        distribute = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=NUM_GPUS_PER_WORKER)
    else:
        distribute = None

    return distribute


def main():

    # Set TensorFlow Hub cache directory to environment variable
    if TFHUB_DIR:
        os.environ['TFHUB_CACHE_DIR'] = TFHUB_DIR

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info(tf.__version__)

    params = {
        'optimizer': tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    }

    # Select DistributeStrategy
    distribute = select_distribute_strategy(DISTRIBUTE_STRATEGY)

    # Show TF_CONFIG
    tf_conf = json.loads(os.environ.get('TF_CONFIG', '{}'))
    tf.logging.info('TF_CONFIG: {}'.format(json.dumps(tf_conf, indent=2)))

    # NOTE:
    # `allow_soft_placement=True` is required because
    # * TensorFlow Hub automatically add operations which run only on CPU such as save
    # * DistributeStrategy use `tf.device('.../GPU:0')` context
    session_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
    )

    config = tf.estimator.RunConfig(
        save_summary_steps=SAVE_STEPS,
        save_checkpoints_steps=SAVE_STEPS,
        keep_checkpoint_max=10,
        train_distribute=distribute,
        session_config=session_config,
    )

    tf.logging.info('Cluster Spec: {}'.format(config.cluster_spec.as_dict()))

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=MODEL_DIR,
        config=config,
        params=params,
    )

    # SessionRunHooks for training
    profiler_hook = tf.train.ProfilerHook(
        save_steps=SAVE_STEPS,
        output_dir=os.path.join(estimator.model_dir, 'timeline')
    )
    train_hooks = [profiler_hook]

    train_spec = tf.estimator.TrainSpec(
        input_fn=pipeline.create_train_input_fn(tfds_dir=TFDS_DIR, batch_size=BATCH_SIZE),
        max_steps=MAX_STEPS,
        hooks=train_hooks,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=pipeline.create_eval_input_fn(tfds_dir=TFDS_DIR, batch_size=BATCH_SIZE),
        steps=None,
        start_delay_secs=10,
        throttle_secs=30,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
