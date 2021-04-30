import os
import random
from typing import Tuple


import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

from config import IMAGES, TRIMAPS, BATCH_SIZE, INPUT_SHAPE, PROPORTION_TEST_IMAGES, EPOCHS
from models import ModelSegmentaions
from data_generator import DataGenerator


class TrainModel():
    def __init__(self, data_images: str = IMAGES, data_trimaps: str = TRIMAPS, batch_size: int = BATCH_SIZE,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE,
                 proportion_test_images: float = PROPORTION_TEST_IMAGES) -> None:
        self.data_images = data_images
        self.data_trimaps = data_trimaps
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.img_size = (image_shape[0], image_shape[1])
        self.proportion_test_images = proportion_test_images

    def train(self) -> None:

        val_samples = int(len(self.data_images) * self.proportion_test_images)
        # random.Random(1337).shuffle(self.data_images)
        # random.Random(1337).shuffle(self.data_trimaps)
        train_input_img_paths = self.data_images[:-val_samples]
        train_target_img_paths = self.data_trimaps[:-val_samples]
        val_input_img_paths = self.data_images[-val_samples:]
        val_target_img_paths = self.data_trimaps[-val_samples:]

        train_gen = DataGenerator(data_images=train_input_img_paths, data_trimaps=train_target_img_paths,
                                  batch_size=self.batch_size, image_shape=self.image_shape)

        val_gen = DataGenerator(data_images=val_input_img_paths, data_trimaps=val_target_img_paths,
                                batch_size=self.batch_size, image_shape=self.image_shape)

        model = ModelSegmentaions().build()
        model.compile(
                        'Adam',
                        loss=sm.losses.bce_jaccard_loss,
                        metrics=[sm.metrics.iou_score],
                      )
        model.fit(train_gen, validation_data=val_gen, batch_size=BATCH_SIZE, epochs=EPOCHS)


x = TrainModel().train()

