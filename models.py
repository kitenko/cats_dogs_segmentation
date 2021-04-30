from typing import Tuple

import segmentation_models as sm

from config import BACKBONE, NUMBER_CLASSES, ACTIVATION, INPUT_SHAPE, ENCODER_WEIGHTS


class ModelSegmentaions():
    def __init__(self, bakebone: str = BACKBONE, num_classes: int = NUMBER_CLASSES, activation: str = ACTIVATION,
                 image_shape: Tuple[int, int, int] = INPUT_SHAPE, encoder_weights: str = ENCODER_WEIGHTS) -> None:
        """

        :param bakebone:
        :param num_classes:
        :param activation:
        :param image_shape:
        :param encoder_weights:
        """
        self.bakebone = bakebone
        self.num_classes = num_classes
        self.activation = activation
        self.image_shape = image_shape
        self.encoder_weights = encoder_weights

    def build(self):
        model = sm.Unet(backend=self.bakebone, encoder_weights=self.encoder_weights, classes=self.num_classes,
                        activation=self.activation, input_shape=self.image_shape)
        return model

