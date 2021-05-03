import os
import json
import random

from config import JSON_FILE_PATH, IMAGES_PATH, TRIMAPS_PATH, PROPORTION_TEST_IMAGES


def prepare_data(json_file_path: str = JSON_FILE_PATH, images_path: str = IMAGES_PATH,
                 trimaps_path: str = TRIMAPS_PATH, proportion_test_images: float = PROPORTION_TEST_IMAGES):

    # reading and shuffling files
    count_images = os.listdir(images_path)
    shuffle_images = random.sample(count_images, len(count_images))

    # create dictionary
    train_test_json = {'train': {'images': {}, 'trimaps': {}}, 'test': {'images': {}, 'trimaps': {}}}

    # checking trimap file in folder
    def find(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(name)

    # filling in dictionary for json file
    for j, i in enumerate(shuffle_images):
        trimaps = find(i.rsplit(".", 1)[0] + '.png', trimaps_path)
        if j < len(shuffle_images) * proportion_test_images:
            train_test_json['test']['images'][j] = os.path.join(images_path, i)
            train_test_json['test']['trimaps'][j] = os.path.join(trimaps_path, trimaps)
        else:
            train_test_json['train']['images'][j] = os.path.join(images_path, i)
            train_test_json['train']['trimaps'][j] = os.path.join(trimaps_path, trimaps)
        print(j)

    # write json file
    with open(json_file_path, 'w') as f:
        json.dump(train_test_json, f, indent=4)


if __name__ == '__main__':
    prepare_data()
