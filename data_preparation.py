import os
import json

import cv2
import numpy as np

from config import JSON_FILE_PATH, IMAGES_PATH, MASKS_PATH, PROPORTION_TEST_IMAGES, LIST_FILE_PATH


def prepare_data(masks_path: str = MASKS_PATH, proportion_test_images: float = PROPORTION_TEST_IMAGES,
                 json_file_path: str = JSON_FILE_PATH, images_path: str = IMAGES_PATH,
                 list_file_path: str = LIST_FILE_PATH) -> None:
    """
    This function creates json file that consist of train and test proportion images.

    :param masks_path: this is pass for masks files.
    :param proportion_test_images: proportion of test images.
    :param json_file_path: path to save json file.
    :param images_path: this is path for image files.
    :param list_file_path: this is path for list file.
    """
    # read list.txt with labels.
    list_txt = {}
    with open(list_file_path) as f:
        lists = f.readlines()
        for i in lists:
            i.split()
            list_txt[i[0]] = i[1:]

    # reading and shuffling files
    images = os.listdir(images_path)
    np.random.shuffle(images)

    # create dictionary
    train_test_json = {'train': [], 'test': []}

    # filling in dictionary for json file
    for j, i in enumerate(images):
        # check file
        if not os.path.exists(os.path.join(masks_path, i.rsplit(".", 1)[0] + '.png')):
            print('no mask for ', i)
            continue
        else:
            masks = i.rsplit(".", 1)[0] + '.png'
            # label = list_txt[i.rsplit(".", 1)[0]]
            label = 1
            img_dict = {'image_path': os.path.join(images_path, i), 'mask_path': os.path.join(masks_path, masks),
                        'class_index': label}
            if cv2.imread(os.path.join(images_path, i)) is None:
                print('broken image')
                continue
            elif j < len(images) * proportion_test_images:
                train_test_json['test'].append(img_dict)
            else:
                train_test_json['train'].append(img_dict)

    # write json file
    with open(json_file_path, 'w') as f:
        json.dump(train_test_json, f, indent=4)


if __name__ == '__main__':
    prepare_data()
