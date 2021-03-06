import json
from typing import Tuple

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from tensorflow import keras

from config import BATCH_SIZE, INPUT_SHAPE_IMAGE, JSON_FILE_PATH, NUMBER_CLASSES, USE_AUGMENTATION


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size: int = BATCH_SIZE, image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE,
                 json_path: str = JSON_FILE_PATH, is_train: bool = False, number_classes: int = NUMBER_CLASSES,
                 use_augmentation_data: bool = USE_AUGMENTATION) -> None:
        """
        Data generator for the task of semantic segmentation cats and dogs.

        :param batch_size: number of images in one batch.
        :param image_shape: this is image shape (height, width, channels).
        :param json_path: this is path for json file.
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        :param number_classes = number of classes.
        :param use_augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
        """

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.number_classes = number_classes
        self.use_augmentation_data = use_augmentation_data

        # read json
        with open(json_path) as f:
            self.data = json.load(f)

        # augmentation data
        if is_train:
            self.data = self.data['train']
            augmentation = images_augmentation(train_data=self.use_augmentation_data)
        else:
            self.data = self.data['test']
            augmentation = images_augmentation(train_data=False)

        self.aug = augmentation
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.

        """
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size + (len(self.data) % self.batch_size != 0)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function makes batch.

        :param batch_idx: batch number.
        :return: image tensor and list with labels tensors for each output.
        """
        batch = self.data[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        masks = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[1], self.number_classes))
        for i, image_dict in enumerate(batch):
            img = cv2.imread(image_dict['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_image = cv2.imread(image_dict['mask_path'], 0)
            augmented = self.aug(image=img, mask=mask_image)
            img = augmented['image']
            mask_image = augmented['mask']
            images[i, :, :, :] = img
            # 3 - object outline, 1 - object, 2 - background
            masks[i, :, :, 0] = np.where(np.logical_or(mask_image == 3, mask_image == 1), 1, 0)
            masks[i, :, :, 1] = np.where(mask_image == 2, 1, 0)
        images = image_normalization(images)
        return images, masks

    def show(self) -> None:
        """
        This function shows image and masks.
        """
        for i in range(len(self)):
            batch = self[i]

            images, masks = batch[0], batch[1]
            fontsize = 8
            for i, j in enumerate(images):
                mask_background = masks[i, :, :, 1]
                mask_animal = masks[i, :, :, 0]
                plt.figure(figsize=[10, 10])
                f, ax = plt.subplots(3, 1)
                ax[0].imshow(j)
                if self.use_augmentation_data is True:
                    ax[0].set_title('Augmented image', fontsize=fontsize)
                else:
                    ax[0].set_title('Original image', fontsize=fontsize)
                ax[1].imshow(mask_animal)
                ax[1].set_title('Mask dog or cat', fontsize=fontsize)
                ax[2].imshow(mask_background)
                ax[2].set_title('Mask background', fontsize=fontsize)
                if plt.waitforbuttonpress(0):
                    plt.close('all')
                    raise SystemExit
                plt.close()


def images_augmentation(train_data: bool = False) -> A.Compose:
    """
    This function makes augmentation data.

    :param train_data: if this parameter is True then augmentation is applied to train dataset.
    :return: augment data
    """
    if train_data is True:
        aug = A.Compose([
            A.Resize(height=INPUT_SHAPE_IMAGE[1], width=INPUT_SHAPE_IMAGE[0]),
            A.Blur(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False,
                                 p=0.5)
            ])
    else:
        aug = A.Compose([A.Resize(height=INPUT_SHAPE_IMAGE[1], width=INPUT_SHAPE_IMAGE[0])])

    return aug


def image_normalization(image: np.ndarray) -> np.ndarray:
    """
    Image normalization.
    :param image: image numpy array.
    :return: normalized image.
    """
    return image / 255.0


if __name__ == '__main__':
    x = DataGenerator()
    x.show()
