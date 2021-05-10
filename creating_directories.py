import os

from config import NAME_MODEL, ENCODER_WEIGHTS, LOGS_DIR, SAVE_MODELS, TENSORBOARD_LOGS, BACKBONE


def create_dirs() -> None:
    os.makedirs(TENSORBOARD_LOGS, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SAVE_MODELS, exist_ok=True)

    try:
        os.makedirs(LOGS_DIR + '/' + NAME_MODEL + BACKBONE + ENCODER_WEIGHTS, exist_ok=True)
    except TypeError:
        os.makedirs(LOGS_DIR + '/' + NAME_MODEL + BACKBONE + 'weights_no', exist_ok=True)
    try:
        os.makedirs(SAVE_MODELS + '/' + NAME_MODEL + BACKBONE + ENCODER_WEIGHTS, exist_ok=True)
    except TypeError:
        os.makedirs(SAVE_MODELS + '/' + NAME_MODEL + BACKBONE + 'weights_no', exist_ok=True)


if __name__ == '__main__':
    create_dirs()
