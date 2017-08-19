"""Helper functions for data loading"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import pandas as pd

FER2013_LABEL_TO_STRING_DICT = {0:'angry',1:'disgust',2:'fear',3:'happy',
                                4:'sad',5:'surprise',6:'neutral'}


def gen_fer2013_csv(csv_path, reshape_width=48, reshape_height=48):
    data = pd.read_csv(csv_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),
                          (reshape_width, reshape_height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    return faces, emotions


def split_data_set(images, labels, split_frac=.2):
    n_train = int((1 - split_frac) * images.shape[1])
    train_images, train_labels = images[:n_train], labels[:n_train]
    validation_images, validation_labels = images[n_train:], labels[n_train:]
    return (train_images, train_labels), (validation_images, validation_labels)


def clean_image(image):
     # preprocess data
    image = image.astype(np.float32)
    image /= 255.0
    image -= 0.5
    image *= 2.0
    return image
