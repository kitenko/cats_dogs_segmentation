import os

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping

from config import JSON_FILE_PATH, EPOCHS, SAVE_CURRENT_MODEL, FULL_NAME_MODEL, TENSORBOARD_LOGS, NUMBER_CLASSES
from models import ModelSegmentaions
from data_generator import DataGenerator
from metrics import Recall, Precision, F1Score
from creating_directories import create_dirs


# create dirs for save logs and models
create_dirs()


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.005
    if epoch > 10:
        learning_rate = 0.001
    if epoch > 20:
        learning_rate = 0.0001
    if epoch > 40:
        learning_rate = 0.00001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def train(dataset_path_json: str) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    :param save_path: path to save weights and training logs.
    """

    train_data_gen = DataGenerator(json_path=dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(json_path=dataset_path_json, is_train=False)

    model = ModelSegmentaions().build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=['accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
                           sm.metrics.f1_score]
                  )
    model.summary()
    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(SAVE_CURRENT_MODEL, FULL_NAME_MODEL + '.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='F1_score',
        mode='max',
        save_best_only=True
    )

    tensor_board = tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGS, update_freq='batch')
    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                        callbacks=[early, model_checkpoint_callback, tensor_board])


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    train(JSON_FILE_PATH)

