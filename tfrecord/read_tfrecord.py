import tensorflow as tf


def _parse(example):
    keys_to_features = {
        # 'x': tf.VarLenFeature(tf.float32),
        'x': tf.VarLenFeature(tf.int64),
        'y': tf.FixedLenFeature((), tf.string),
    }
    parsed = tf.parse_single_example(example, keys_to_features)
    parsed['x'] = tf.sparse_tensor_to_dense(parsed['x'])
    return parsed


ds = tf.data.TFRecordDataset(['sample-00000-of-00001.tfrecord'])
ds = ds.map(_parse)
# NOTE:
# * can not use `batch_size >= 2` because of variable length
# * need zero padding if you want to use `batch_size >= 2`
ds = ds.batch(1)

iterator = ds.make_one_shot_iterator()
next_elements = iterator.get_next()

with tf.Session() as sess:
    samples = sess.run(next_elements)

print(samples)
