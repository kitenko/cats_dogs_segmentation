import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import build_model
from config import INPUT_SHAPE_IMAGE


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    return parser.parse_args()


def preparing_frame(image: np.ndarray, model) -> np.ndarray:
    """
    This function prepares the image and calls the predicted method.

    :param image: this is input image or frame.
    :param model: assembled model with loaded weights.
    :return: image with an overlay mask
    """
    image = cv2.resize(image, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
    plt.figure(figsize=(20, 20))
    mask = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    mask = np.where(mask >= 0.5, 1, 0)[:, :, 0]
    image[:, :, 2] = np.where(mask == 1, 100, image[:, :, 2])
    return image


def visualization() -> None:
    """
    This function captures webcam video and resizes the image.
    """
    args = parse_args()
    model = build_model()
    model.load_weights(args.weights)

    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.resize(frame, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
        predict_mask = preparing_frame(image=frame, model=model)
        predict_mask = cv2.resize(predict_mask, (720, 720))
        cv2.imshow('frame', predict_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    visualization()
