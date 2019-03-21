import tensorflow as tf
from . import model, pipeline


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('max_steps', 10000, 'maximum global step for training')
tf.flags.DEFINE_integer('save_steps', 2000, 'interval of saving summaries and checkpoints')
tf.flags.DEFINE_string('model_dir', './outputs', 'where to save models')

MAX_STEPS = FLAGS.max_steps
SAVE_STEPS = FLAGS.save_steps
MODEL_DIR = FLAGS.model_dir


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        'optimizer': tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    }

    # distribute = tf.contrib.distribute.MirroredStrategy()

    tf.logging.info('create config')
    config = tf.estimator.RunConfig(
        save_summary_steps=SAVE_STEPS,
        save_checkpoints_steps=SAVE_STEPS,
        keep_checkpoint_max=10,
        # train_distribute=distribute,
    )

    tf.logging.info('create estimator')
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=MODEL_DIR,
        config=config,
        params=params,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=pipeline.create_train_input_fn(tfds_dir=TFDS_DIR),
        max_steps=MAX_STEPS,
        hooks=None,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=pipeline.create_eval_input_fn(tfds_dir=TFDS_DIR),
        steps=None,
        start_delay_secs=0,
        throttle_secs=0,
    )

    tf.logging.info('train and evaluate')
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    import tensorflow_datasets as tfds
    # ds = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir=TFDS_DIR)
    tf.logging.info('create mnist builder')
    mnist_builder = tfds.builder('mnist')
    tf.logging.info('download and prepare with mnist builder')
    mnist_builder.download_and_prepare()
    tf.logging.info('construct a tf.data.Dataset')
    ds = mnist_builder.as_dataset(split=tfds.Split.TRAIN)
    # tf.logging.info('hello')
    # main()
