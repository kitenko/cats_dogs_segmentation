import os

import tensorflow as tf
import segmentation_models as sm

from config import JSON_FILE_PATH, BATCH_SIZE, PROPORTION_TEST_IMAGES, EPOCHS
from models import ModelSegmentaions
from data_generator import DataGenerator


def train(dataset_path_json: str) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    :param save_path: path to save weights and training logs.
    """

    train_data_gen = DataGenerator(dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(dataset_path_json, is_train=False)

    model = ModelSegmentaions().build()
    model.compile('Adam',
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score]
                  )
    model.summary()

    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8)


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    train(JSON_FILE_PATH)