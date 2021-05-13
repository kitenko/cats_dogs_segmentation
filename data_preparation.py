import os
import json
import random

import cv2

from config import JSON_FILE_PATH, IMAGES_PATH, MASKS_PATH, PROPORTION_TEST_IMAGES


def prepare_data(trimaps_path: str = MASKS_PATH, proportion_test_images: float = PROPORTION_TEST_IMAGES,
                 json_file_path: str = JSON_FILE_PATH, images_path: str = IMAGES_PATH) -> None:
    """
    This function creates json file that consist of train and test proportion images.

    :param trimaps_path: this is pass for trimap files.
    :param proportion_test_images: proportion of test images.
    :param json_file_path: path to save json file.
    :param images_path: this is path for image files.
    """
    # read list.txt with labels.
    list_txt = {}
    with open('data/annotations/list.txt') as f:  # path
        while True:
            line = f.readline() # f.readlines() and for
            line = line.split()
            if not line:
                break
            list_txt[line[0]] = line[1:]

    # reading and shuffling files
    count_images = os.listdir(images_path)
    import numpy as np
    np.random.shuffle(count_images)
    # shuffle_images = random.sample(count_images, len(count_images))

    # create dictionary
    train_test_json = {'train': [], 'test': []}

    def find(name, path) -> str:
        """
        This function checking trimap file in folder.

        :param name: The name of the file to find.
        :param path: This is the path where look for the file.
        :return: Name trimaps file for image.
        """
        for root, dirs, files in os.walk(path):
            if name in files:
                return name

    # filling in dictionary for json file
    for j, i in enumerate(shuffle_images):
        if not os.path.exists('some/path'):
            continue
        try:
            trimaps = find(i.rsplit(".", 1)[0] + '.png', trimaps_path)
            # label = list_txt[i.rsplit(".", 1)[0]]
            label = 1
            img_dict = {'image_path': os.path.join(images_path, i), 'mask_path': os.path.join(trimaps_path, trimaps),
                        'class_index': label}
            if cv2.imread(os.path.join(images_path, i)) is None:
                print('broken image')
                continue
            elif j < len(shuffle_images) * proportion_test_images:
                train_test_json['test'].append(img_dict)
            else:
                train_test_json['train'].append(img_dict)
        except KeyError:
            print(' no mask for ', i)

    # write json file
    with open(json_file_path, 'w') as f:
        json.dump(train_test_json, f, indent=4)


if __name__ == '__main__':
    prepare_data()
