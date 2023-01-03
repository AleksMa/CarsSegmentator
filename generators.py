import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence
import numpy as np

from config import *


GRAYSCALE_COLORS = [
    0.,
    255.,
    200.,
    150.,
    100.,
]


class TrainGenerator(Sequence):
    def __init__(self):
        self.data_list = os.listdir(IMG_TRAIN_DIR)

    def __len__(self):
        return len(self.data_list) // BATCH_SIZE

    def __getitem__(self, item):
        i = item * BATCH_SIZE
        batch_input_data = self.data_list[i:i + BATCH_SIZE]
        x = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype='float32')
        y = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype='float32')

        for j, path in enumerate(batch_input_data):
            img = load_img(path=IMG_TRAIN_DIR + path,
                           target_size=(IMG_SIZE, IMG_SIZE),
                           color_mode='grayscale')

            label = load_img(path=LABLE_TRAIN_DIR + path[:-4] + '.png',
                             target_size=(IMG_SIZE, IMG_SIZE),
                             color_mode='grayscale')

            x[j] = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
            y[j] = np.reshape(label, (IMG_SIZE, IMG_SIZE, 1))

        x = x / 255.
        y = y[:, :, :, 0] if (len(y.shape) == 4) else y[:, :, 0]

        y[(y != GRAYSCALE_COLORS[0]) &
          (y != GRAYSCALE_COLORS[1]) &
          (y != GRAYSCALE_COLORS[2]) &
          (y != GRAYSCALE_COLORS[3]) &
          (y != GRAYSCALE_COLORS[4])
          ] = 0.

        z = np.zeros(y.shape + (CLASSES,))
        for j, color in enumerate(GRAYSCALE_COLORS):
            z[y == color, j] = 1

        return x, z


class TestGenerator(Sequence):
    def __init__(self):
        self.data_list = os.listdir(IMG_TEST_DIR)

    def __len__(self):
        return len(self.data_list) // BATCH_SIZE

    def __getitem__(self, item):
        i = item * BATCH_SIZE
        batch_input_data = self.data_list[i:i + BATCH_SIZE]
        x = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype='float32')

        for j, path in enumerate(batch_input_data):
            img = load_img(path=IMG_TEST_DIR + path,
                           target_size=(IMG_SIZE, IMG_SIZE),
                           color_mode='grayscale')

            x[j] = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
            x[j] = x[j] / 255
        return x
