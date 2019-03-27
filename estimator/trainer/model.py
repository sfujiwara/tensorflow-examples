import tensorflow as tf
import tensorflow_hub as hub


def model_fn(features, labels, mode, params):

    n_classes = 2

    # Extract inputs
    x = features

    # Build ResNet
    module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1', trainable=True)
    x = module(x)

    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    logits = tf.layers.dense(x, n_classes, activation=None)

    # Build loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Build training operation
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        train_op = params['optimizer'].minimize(loss, global_step)
    else:
        train_op = None

    # Build eval metric operations
    classes = tf.argmax(logits, axis=1)
    probabilities = tf.nn.softmax(logits)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=classes)
    }

    # Build EstimatorSpec
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )

    return estimator_spec
