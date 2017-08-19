"""Test ResXceptionNet on Image"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from utils import clean_image, FER2013_LABEL_TO_STRING_DICT

from keras.models import load_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--weights_path", type=str, help="path to weights")
    args = parser.parse_args()

    net = load_model(args.weights_path, compile=False)
    net_input_shape = net.input_shape[1:3]

    image = cv2.imread(args.image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (net_input_shape))
    gray_image = clean_image(gray_image)
    gray_image = np.expand_dims(gray_image, 0)
    gray_image = np.expand_dims(gray_image, -1)

    emotion_proba = net.predict(gray_image)

    def proba2map(proba):
        assert proba.size == 7
        proba_map = {}
        for i in range(7):
            proba_map[FER2013_LABEL_TO_STRING_DICT[i]] = proba[i]

    emotion_map = proba2map(emotion_proba)
    print(emotion_map)


