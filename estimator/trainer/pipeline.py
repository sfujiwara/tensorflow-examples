import tensorflow as tf
import tensorflow_datasets as tfds


def train_input_fn():
    ds = tfds.load('mnist', split=tfds.Split.TRAIN)
    ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32)/255., x['label']))
    ds = ds.repeat().shuffle(1024).batch(32)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def eval_input_fn():
    ds = tfds.load('mnist', split=tfds.Split.TEST)
    ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32)/255., x['label']))
    ds = ds.repeat(1).batch(100)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
