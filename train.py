import os

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping

from models import build_model
from data_generator import DataGenerator
from config import (JSON_FILE_PATH, EPOCHS, SAVE_CURRENT_MODEL, TENSORBOARD_LOGS, MODELS_DATA, MODEL_NAME, BACKBONE,
                    SAVE_MODELS, LOGS_DIR_CURRENT_MODEL, SAVE_CURRENT_TENSORBOARD_LOGS, LEARNING_RATE)


def train(dataset_path_json: str = JSON_FILE_PATH) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    """
    # create dirs
    for p in [TENSORBOARD_LOGS, MODELS_DATA, SAVE_MODELS, SAVE_CURRENT_MODEL, LOGS_DIR_CURRENT_MODEL,
              SAVE_CURRENT_TENSORBOARD_LOGS]:
        os.makedirs(p, exist_ok=True)

    train_data_gen = DataGenerator(json_path=dataset_path_json, is_train=True)
    test_data_gen = DataGenerator(json_path=dataset_path_json, is_train=False)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=['accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
                           sm.metrics.f1_score]
                  )
    model.summary()
    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(SAVE_CURRENT_MODEL, MODEL_NAME + BACKBONE + '.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='iou_score',
        mode='max',
        save_best_only=True
    )

    tensor_board = tf.keras.callbacks.TensorBoard(os.path.join(TENSORBOARD_LOGS, MODEL_NAME + BACKBONE),
                                                  update_freq='batch')
    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                        callbacks=[early, model_checkpoint_callback, tensor_board])


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    train()
