import os
from datetime import datetime

date_time_for_save = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

BATCH_SIZE = 8
NUMBER_CLASSES = 1 + 1
INPUT_SHAPE_IMAGE = (256, 256, 3)
PROPORTION_TEST_IMAGES = 0.2
EPOCHS = 150
LEARNING_RATE = 0.0001

BACKBONE = 'resnet18'
AUGMENTATION_DATA = True
# ENCODER_WEIGHTS = None
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax'
NAME_MODEL = 'Unet'
FULL_NAME_MODEL = NAME_MODEL + BACKBONE

DATA_PATH = 'data'
JSON_FILE_PATH = os.path.join(DATA_PATH, 'data.json')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'annotations')
MASKS_PATH = os.path.join(ANNOTATIONS_PATH, 'masks')
LIST_FILE_PATH = os.path.join(ANNOTATIONS_PATH, 'list.txt')
PATH_FOR_MODEL_WEIGHTS = 'models_data/save_models/Unet_imagenet_2021-05-13 16:31:31/Unetresnet18.h5'

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

LOGS_DIR_CURRENT_MODEL = os.path.join(LOGS, NAME_MODEL + '_' + str(ENCODER_WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, NAME_MODEL + '_' + str(ENCODER_WEIGHTS) + '_' + date_time_for_save)
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, NAME_MODEL + '_' + str(ENCODER_WEIGHTS) + '_' +
                                             date_time_for_save)
