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
    ds = ds.repeat().batch(100)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.75),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        # metrics=['accuracy']
    )

    return model


def main():
    model = create_model()

    config = tf.estimator.RunConfig(
        model_dir='./outputs',
        train_distribute=tf.distribute.MirroredStrategy(),
    )

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        config=config,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=100,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
