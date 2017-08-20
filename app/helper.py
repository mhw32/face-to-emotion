"""helper.py: Utility functions to crop images
and prepare for input into the net.
"""

import base64
import numpy as np

import cv2
from io import StringIO

from app import nn
from model.utils import (clean_image,
                         FER2013_LABEL_TO_STRING_DICT)


def read_base64_image(base64_str):
    """Converts base64 string to numpy array.

    @param base64_str: base64 encoded string
    @return: numpy array RGB (H x W x C)
    """
    nparr = np.fromstring(base64_str.decode('base64'),
                          np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def image_to_face_crop(rgb_image, trueface_responses):
    """Crops image based on trueface coordinates

    @param rgb_image: numpy array HxWxC
    @param trueface_responses: JSON/dict from trueface API
    @return: list of numpy array RGB (arbitrary sized)
    """
    face_images = []
    for response in trueface_responses:
        bbox = response['bounding_box'][:-1]  # ignore confidence
        x0, x1 = int(np.round(bbox[1])),  int(np.round(bbox[3]))
        y0, y1 = int(np.round(bbox[0])), int(np.round(bbox[2]))
        face_image = rgb_image[x0:x1, y0:y1]
        face_images.append(face_image)

    return face_images


def index_to_emotion(proba):
    """Return interpretable emotion instead of index

    @param proba: numpy array of size 7
    @return: map of emotion string to proba
    """
    assert proba.size == 7
    proba_map = {}
    for i in range(7):
        proba_map[FER2013_LABEL_TO_STRING_DICT[i]] = proba[i]

    return proba_map


def classify_image(image):
    """Call ResXception to return emotion probabilities
    from Keras model.

    @param image: RGB cropped face image
    @return: map from emotion to probability
    """
    nn_input_shape = nn.input_shape[1:3]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (nn_input_shape))
    gray_image = clean_image(gray_image)
    gray_image = np.expand_dims(gray_image, 0)
    gray_image = np.expand_dims(gray_image, -1)

    emotion_proba = nn.predict(gray_image)
    emotion_map = index_to_emotion(emotion_proba)
    return emotion_map

