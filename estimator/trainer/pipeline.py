import tensorflow as tf
import tensorflow_datasets as tfds


def create_train_input_fn(tfds_dir):

    def train_input_fn():
        ds = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir=tfds_dir)
        ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32)/255., x['label']))
        ds = ds.repeat().shuffle(1024).batch(32)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    return train_input_fn


def create_eval_input_fn(tfds_dir):

    def eval_input_fn():
        ds = tfds.load('mnist', split=tfds.Split.TEST, data_dir=tfds_dir)
        ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32)/255., x['label']))
        ds = ds.repeat(1).batch(100)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    return eval_input_fn


def serving_input_receiver_fn():
    return
