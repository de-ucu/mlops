import os
import neptune
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from neptunecontrib.monitoring.keras import NeptuneMonitor

SEED = 1
BATCH_SIZE = 10
DATA_ROOT = Path("/", "data", "pets", '01-catdog', '02-tfrecords')
tf.keras.backend.set_image_data_format('channels_last')


neptune.init('eawer/animal-classifier')
neptune_experiment = neptune.create_experiment(name='InceptionResNetV2')


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)


features = {
    'label': tf.io.FixedLenFeature([], dtype=tf.int64),
    'image': tf.io.FixedLenFeature([], dtype=tf.string),
}

def parse_example(example_proto):
    return tf.io.parse_example(example_proto, features)

def parse_record(example):
    image = tf.io.decode_raw(example["image"], tf.uint8)
    image = tf.reshape(image, (-1, 299, 299, 3))
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example["label"]

    return tf.keras.applications.inception_resnet_v2.preprocess_input(image), label


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
  layers.experimental.preprocessing.RandomContrast(0.1),
])

def prepare_dataset(paths, batch_size, augment=False, seed=SEED):
    ds = tf.data.TFRecordDataset(paths, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP")
    ds = ds.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(parse_example)
    ds = ds.batch(batch_size)
    ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()

    return ds.prefetch(tf.data.AUTOTUNE)


train_paths = [str(p) for p in Path(DATA_ROOT, "train").glob("*.tfrecord")]
train_dataset = prepare_dataset(train_paths, BATCH_SIZE, augment=True)

test_paths = [str(p) for p in Path(DATA_ROOT, "test").glob("*.tfrecord")]
test_dataset = prepare_dataset(test_paths, BATCH_SIZE)


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.applications.InceptionResNetV2(
        input_shape=(299, 299, 3),
        classes=2,
        weights=None,
        classifier_activation="softmax",
    )

    model.compile(
        optimizer="adam",
        loss=[tf.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy", "mse"]
    )


callbacks = [
    tf.keras.callbacks.LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.01, 10, alpha=0.0001)),
    NeptuneMonitor(),
]

model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=callbacks,
)


