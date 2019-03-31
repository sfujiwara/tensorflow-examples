import multiprocessing
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


def create_train_input_fn(tfds_dir, batch_size, tfds_dataset):

    def train_input_fn():
        ds = tfds.load(tfds_dataset, split=tfds.Split.TRAIN, data_dir=tfds_dir)
        # NOTE:
        # * `shuffle_and_repeat` has higher performance than using `shuffle` and `repeat`
        # * https://www.tensorflow.org/guide/performance/datasets#repeat_and_shuffle
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=32))
        # NOTE:
        # * `map_and_batch` has higher performance than using `map` and `batch`
        # * Use the number of CPUs or `tf.data.experimental.AUTOTUNE` for `num_parallel_calls`
        # * https://www.tensorflow.org/guide/performance/datasets#map_and_batch
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                map_func=preprocess,
                batch_size=batch_size,
                num_parallel_calls=multiprocessing.cpu_count(),
            )
        )
        ds = ds.prefetch(1)
        return ds

    return train_input_fn


def create_eval_input_fn(tfds_dir, batch_size, tfds_dataset):

    if tfds_dataset == 'imagenet2012':
        split = tfds.Split.VALIDATION
    else:
        split = tfds.Split.TEST

    def eval_input_fn():
        ds = tfds.load(tfds_dataset, split=split, data_dir=tfds_dir)
        ds = ds.repeat(1)
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                map_func=preprocess,
                batch_size=batch_size,
                num_parallel_calls=multiprocessing.cpu_count(),
            )
        )
        ds = ds.prefetch(1)
        return ds

    return eval_input_fn


def serving_input_receiver_fn():
    return
