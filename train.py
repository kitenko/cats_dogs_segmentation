import os

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping

from models import ModelSegmentaions
from data_generator import DataGenerator
from creating_directories import create_dirs
from config import JSON_FILE_PATH, EPOCHS, SAVE_CURRENT_MODEL, TENSORBOARD_LOGS, FULL_NAME_MODEL_H5, FULL_NAME_MODEL

# create dirs for save logs and models
create_dirs()


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.005
    if epoch > 5:
        learning_rate = 0.001
    if epoch > 15:
        learning_rate = 0.0001
    if epoch > 60:
        learning_rate = 0.00005
    if epoch > 100:
        learning_rate = 0.00001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def train(dataset_path_json: str) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    """

    train_data_gen = DataGenerator(json_path=dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(json_path=dataset_path_json, is_train=False)

    model = ModelSegmentaions().build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=['accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
                           sm.metrics.f1_score]
                  )
    model.summary()
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(SAVE_CURRENT_MODEL, FULL_NAME_MODEL_H5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='iou_score',
        mode='max',
        save_best_only=True
    )

    tensor_board = tf.keras.callbacks.TensorBoard(os.path.join(TENSORBOARD_LOGS, FULL_NAME_MODEL), update_freq='batch')
    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                        callbacks=[early, model_checkpoint_callback, tensor_board, lr_callback])


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    train(JSON_FILE_PATH)
