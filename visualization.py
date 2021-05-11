import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from models import ModelSegmentaions
from config import INPUT_SHAPE_IMAGE

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

images = np.zeros((1, INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0], 3))
model = ModelSegmentaions().build_unet()
model.load_weights('save_models/Unetresnet18imagenet/Unetresnet18.h5')
img = cv2.imread('/home/andre/Загрузки/Telegram Desktop/photo_2021-05-07_16-11-30.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(img, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
resized_image = resized_image / 255.0
images[0, :, :, :] = resized_image
plt.figure(figsize=(20, 20))
mask = model.predict(images)
mask = mask[0]
mask[mask >= 0.5] = 1
mask[mask < 0.5] = 0
mask = np.stack((mask,)*3, axis=-1)
mask = mask[:, :, 2]
images = images.squeeze()

images_mask = cv2.addWeighted(images, alpha=0.5, src2=mask, beta=1, gamma=0)
# show the mask and the segmented image
combined = np.concatenate([images, mask, images_mask], axis=1)
plt.axis('off')
plt.imshow(combined)
plt.show()

plt.show()

# # 2
# model = ModelSegmentaions().build_unet()
# model.load_weights('save_models/Unetresnet34imagenet/Unetresnet34.h5')
# raw = Image.open('/home/andre/Загрузки/Telegram Desktop/photo_2021-05-07_16-11-30.jpg')
# raw = np.array(raw.resize((512, 512)))/255.
#
# #predict the mask
# pred = model.predict(np.expand_dims(raw, 0))
#
# #mask post-processing
# msk = pred.squeeze()
# msk[msk >= 0.5] = 1
# msk[msk < 0.5] = 0
# msk = np.stack((msk,)*3, axis=-1)
# msk = msk[:, :, 0]
#
# #show the mask and the segmented image
# combined = np.concatenate([raw, msk, raw*msk], axis=1)
# plt.axis('off')
# plt.imshow(combined)
# plt.show()
