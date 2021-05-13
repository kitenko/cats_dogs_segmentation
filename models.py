from typing import Tuple

import tensorflow as tf
import segmentation_models as sm

from config import BACKBONE, NUMBER_CLASSES, ACTIVATION, INPUT_SHAPE_IMAGE, ENCODER_WEIGHTS


class ModelSegmentaions():
    def __init__(self, backbone_name: str = BACKBONE, num_classes: int = NUMBER_CLASSES, activation: str = ACTIVATION,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE, encoder_weights: str = ENCODER_WEIGHTS) -> None:
        """
        This class creates a model from segmentation_models library depending on the input parameters.

        :param backbone_name: name of classification model (without last dense layers) used as feature extractor to
                              build segmentation model.
        :param num_classes: number of classes
        :param activation: name of one of keras.activations for last model layer (e.g. sigmoid, softmax, linear).
        :param image_shape: this is image shape (height, width, channels).
        :param encoder_weights: one of None (random initialization), imagenet (pre-training on ImageNet).
        """
        self.bakebone_name = backbone_name
        self.num_classes = num_classes
        self.activation = activation
        self.image_shape = image_shape
        self.encoder_weights = encoder_weights


def build_unet():
    """
    This function builds Unet model based on the input parameters.

    :return: tf.keras.model
    """
    model = sm.Unet(backbone_name=self.bakebone_name, encoder_weights=self.encoder_weights,
                    classes=self.num_classes, activation=self.activation, input_shape=self.image_shape)
    return model

