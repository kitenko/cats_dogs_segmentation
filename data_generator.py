import os
from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from config import IMAGES, TRIMAPS, BATCH_SIZE, INPUT_SHAPE, PROPORTION_TEST_IMAGES


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_images: str = IMAGES, data_trimaps: str = TRIMAPS, batch_size: int = BATCH_SIZE,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE, is_train: bool = True,
                 proportion_test_images: float = PROPORTION_TEST_IMAGES) -> None:

        self.data_images = data_images
        self.data_trimaps = data_trimaps
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.img_size = (image_shape[0], image_shape[1])
        self.proportion_test_images = proportion_test_images

        self.data_images = sorted(
            [
                os.path.join(data_images, fname)
                for fname in os.listdir(data_images)
                if fname.endswith(".jpg")
            ]
        )
        self.data_trimaps = sorted(
            [
                os.path.join(data_trimaps, fname)
                for fname in os.listdir(data_images)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )

    def __len__(self) -> int:
        #return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)
        return len(self.data_images) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.data_images[i: i + self.batch_size]
        batch_target_img_paths = self.data_trimaps[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

