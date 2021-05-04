import json
from typing import Tuple

import cv2
import numpy as np
from tensorflow import keras

from config import BATCH_SIZE, INPUT_SHAPE_IMAGE, PROPORTION_TEST_IMAGES, JSON_FILE_PATH, INPUT_SHAPE_TRIMAPS


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size: int = BATCH_SIZE, image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE,
                 is_train: bool = False, proportion_test_images: float = PROPORTION_TEST_IMAGES,
                 json_path: str = JSON_FILE_PATH, trimaps_shape: Tuple[int, int, int] = INPUT_SHAPE_TRIMAPS) -> None:

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.trimaps_shape = trimaps_shape
        self.proportion_test_images = proportion_test_images

        # read json
        with open(json_path) as f:
            self.data = json.load(f)

        if is_train:
            self.data = self.data['train']
        else:
            self.data = self.data['test']

        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.

        """
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)
        # return len(self.data) // self.batch_size

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function makes batch.

        :param batch_idx: batch number.
        :return: image tensor and list with labels tensors for each output.
        """
        batch = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        masks = np.zeros((self.batch_size, self.trimaps_shape[0], self.trimaps_shape[1], self.trimaps_shape[2]))
        for i, image_dict in enumerate(batch):
            img = cv2.imread(image_dict['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
            images[i, :, :, :] = resized_image
            class_index = int(image_dict['class_index'])
            mask_image = cv2.imread(image_dict['mask_path'], 0)
            mask_image = cv2.resize(mask_image, (self.trimaps_shape[1], self.trimaps_shape[0]))
            masks[i, :, :, class_index] = np.where(mask_image == 3, 1, 0)  # 3 - контур
            masks[i, :, :, class_index] = np.where(mask_image == 1, 1, 0)  # 1 - объект
            masks[i, :, :, -1] = np.where(mask_image == 2, 1, 0)  # 2 - фон
        images = image_normalization(images)
        return images, masks


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.
    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


if __name__ == '__main__':
    x = DataGenerator()
    x.__len__()