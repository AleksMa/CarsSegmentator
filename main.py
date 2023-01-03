import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

from config import *
from generators import *
from model import *


def draw_labels(img):
    img_out = np.zeros(img[:, :, 0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i, j])
            img_out[i, j] = COLORS[index_of_class]
    return img_out


def save_results(save_path, files):
    data = os.listdir(IMG_TEST_DIR)
    for i, item in enumerate(files):
        img = draw_labels(item)
        img = img.astype(np.uint8)
        io.imsave(os.path.join(save_path + data[i][:-4] + ".png"), img)


def prepare_data():
    images = os.listdir(LABLE_TRAIN_DIR)

    for i in tqdm(range(len(images))):
        img = cv2.imread(LABLE_TRAIN_DIR + images[i], cv2.IMREAD_GRAYSCALE)
        for i, color in enumerate(GRAYSCALE_COLORS):
            img[img == i] = int(color)
        cv2.imwrite(LABLE_TRAIN_DIR + images[i], img)

    for image in tqdm(os.listdir(IMG_TRAIN_DIR)):
        try:
            img = cv2.imread(IMG_TRAIN_DIR + image, 1)
            label = cv2.imread(LABLE_TRAIN_DIR + image[:-4] + '.png')

            img = cv2.flip(img, -1)
            label = cv2.flip(label, -1)

            cv2.imwrite(IMG_TRAIN_DIR + image[:-4] + 'flipped.jpg', img)
            cv2.imwrite(LABLE_TRAIN_DIR + image[:-4] + 'flipped.png', label)

            rows, cols, _ = img.shape
            points1 = np.float32([[50, 50], [200, 50], [50, 200]])
            points2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(points1, points2)
            img = cv2.warpAffine(img, M, (cols, rows))
            label = cv2.warpAffine(label, M, (cols, rows))

            cv2.imwrite(IMG_TRAIN_DIR + image[:-4] + 'transformed.jpg', img)
            cv2.imwrite(LABLE_TRAIN_DIR +
                        image[:-4] + 'transformed.png', label)

        except Exception:
            continue


def main():
    disable_eager_execution()

    if GENERATE_EXTRA_SAMPLES:
        prepare_data()

    train_data_gen = TrainGenerator()
    model = get_model((IMG_SIZE, IMG_SIZE))
    fit_dump = model.fit_generator(train_data_gen, epochs=EPOCHS)
    test_data = TestGenerator()
    result = model.predict_generator(test_data, verbose=1)

    save_results(RESULT_DIR, result)

    plt.plot(
        range(
            0, len(
                fit_dump.history['accuracy'])), fit_dump.history['accuracy'])
    plt.title('accuracy')
    plt.show()


if __name__ == "__main__":
    main()
