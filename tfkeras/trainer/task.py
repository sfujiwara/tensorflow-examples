import tensorflow as tf
import tensorflow_datasets as tfds


def train_dataset():
    ds = tfds.load('mnist', split=tfds.Split.TRAIN)
    ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32)/255., x['label']))
    ds = ds.repeat().shuffle(1024).batch(32)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def eval_dataset():
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
    return model


def main():
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.01, nesterov=True),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ds_train = train_dataset()
    ds_eval = eval_dataset()

    model.fit(
        ds_train,
        epochs=5,
        steps_per_epoch=int(60000/32),
        validation_data=ds_eval,
        validation_steps=100
    )


if __name__ == '__main__':
    main()
