import tensorflow as tf


def model_fn(features, labels, mode, params):

    # Extract inputs
    x = features

    # Build network
    x = tf.layers.conv2d(x, 32, [3, 3], padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 64, [3, 3], padding='same', activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[1, 1])
    x = tf.layers.dropout(x, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    x = tf.layers.dropout(x, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(x, 10)

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