import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess(ds):

    # Resize image
    x = tf.image.resize_images(ds['image'], [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Cast to tf.float32
    x = tf.cast(x, tf.float32)
    # Scale image to [0.0, 1.0]
    x = x / 255.

    return x, ds['label']


def create_train_input_fn(tfds_dir, batch_size):

    def train_input_fn():
        ds = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, data_dir=tfds_dir)
        ds = ds.map(preprocess)
        ds = ds.repeat().shuffle(32).batch(batch_size)
        ds = ds.prefetch(1)
        return ds

    return train_input_fn


def create_eval_input_fn(tfds_dir, batch_size):

    def eval_input_fn():
        ds = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, data_dir=tfds_dir)
        ds = ds.map(preprocess)
        ds = ds.repeat(1).batch(batch_size)
        ds = ds.prefetch(1)
        return ds

    return eval_input_fn


def serving_input_receiver_fn():
    return
