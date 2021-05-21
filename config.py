import os
from datetime import datetime

BATCH_SIZE = 8
NUMBER_CLASSES = 2
INPUT_SHAPE_IMAGE = (256, 256, 3)
PROPORTION_TEST_IMAGES = 0.2
EPOCHS = 150
LEARNING_RATE = 0.0001
USE_AUGMENTATION = False

BACKBONE = 'efficientnetb0'
ENCODER_WEIGHTS = 'imagenet'
OUTPUT_ACTIVATION = 'softmax'
MODEL_NAME = 'Linknet'

DATA_PATH = 'data'
JSON_FILE_PATH = os.path.join(DATA_PATH, 'data.json')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'annotations')
MASKS_PATH = os.path.join(ANNOTATIONS_PATH, 'masks')
LIST_FILE_PATH = os.path.join(ANNOTATIONS_PATH, 'list.txt')

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, MODEL_NAME + '_' + str(ENCODER_WEIGHTS) + '_' +
                                  date_time_for_save + '_' + str(USE_AUGMENTATION))
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, MODEL_NAME + '_' + str(ENCODER_WEIGHTS) + '_' +
                                             date_time_for_save + '_' + str(USE_AUGMENTATION))
