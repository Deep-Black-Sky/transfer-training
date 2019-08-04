import cv2
import numpy as np
from keras.datasets import cifar100
from keras.utils import np_utils

def get_datasets(width, height):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # resize
    x_train = np.array([cv2.resize(img, (width, height)) for img in x_train])
    x_test = np.array([cv2.resize(img, (width, height)) for img in x_test])
    # label index to one-hot vector.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)
