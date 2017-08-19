from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from keras.layers import (Activation, Convolution2D, Dropout,
                          Conv2D, AveragePooling2D, BatchNormalization,
                          GlobalAveragePooling2D, MaxPooling2D,
                          SeparableConv2D, Flatten, Input)
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2


def ResXceptionBlock(input, size):
    # residual component
    r = Conv2D(size, (1, 1), strides=(2, 2),
               padding='same', use_bias=False)(input)
    r = BatchNormalization()(r)
    # depth-wise separable conv
    x = SeparableConv2D(size, (3, 3), padding='same',
                        kernel_regularizer=l2(0.01),
                        use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(size, (3, 3), padding='same',
                        kernel_regularizer=l2(0.01),
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # sum the two components
    output = add([x, r])
    return output


def ResXceptionNet(input_shape, n_class):
    input = Input(input_shape)
    # early convolutional layers
    x = Conv2D(8, (3, 3), strides=(1, 1),
               kernel_regularizer=l2(0.01),
               use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1),
               kernel_regularizer=l2(0.01),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # add 4 ResXception blocks of increasing size
    x = ResXceptionBlock(x, 16)
    x = ResXceptionBlock(x, 32)
    x = ResXceptionBlock(x, 64)
    x = ResXceptionBlock(x, 128)
    # finish with a fully convolutional layer
    x = Conv2D(n_class, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    output = Activation('softmax', name='proba')(x)
    return Model(input, output)
