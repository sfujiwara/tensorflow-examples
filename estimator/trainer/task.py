import tensorflow as tf
from . import model, pipeline


MAX_STEPS = 10000
SAVE_STEPS = 2000


def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    params = {
        'optimizer': tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    }

    config = tf.estimator.RunConfig(
        save_summary_steps=SAVE_STEPS,
        save_checkpoints_steps=SAVE_STEPS,
        keep_checkpoint_max=None,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir='./outputs',
        config=config,
        params=params,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=pipeline.train_input_fn,
        max_steps=MAX_STEPS,
        hooks=None,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=pipeline.eval_input_fn,
        steps=None,
        start_delay_secs=0,
        throttle_secs=0,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
