"""Train ResXceptionNet on FER2013"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from resXception import ResXceptionNet
from utils import (gen_fer2013_csv, split_data_set,
                   clean_image)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to fer2013 dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--split_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--anneal_lr", type=float, default=0.1)
    args = parser.parse_args()

    # data augmentation
    data_augmentor = ImageDataGenerator(featurewise_center=False,
                                        featurewise_std_normalization=False,
                                        rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=.1,
                                        horizontal_flip=True)

    # load net architecture
    net = ResXceptionNet((64, 64, 1), 7)
    net.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

    # define optimizer params
    early_stop = EarlyStopping('val_loss', patience=args.patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=args.anneal_lr,
                                  patience=int(args.patience / 4))

    X, y = gen_fer2013_csv(args.data_path)
    # preprocess data
    X = clean_image(X)

    # split into training and validation sets
    (X_train, y_train), (X_validation, y_validation) = split_data_set(X, y, args.split_frac)
    n_train = X_train.shape[0]

    # call fit on net
    net.fit_generator(data_augmentor.flow(X_train, y_train, args.batch_size),
                      steps_per_epoch=int(n_train / batch_size),
                      epochs=args.epochs, verbose=1,
                      callbacks=[early_stop, reduce_lr],
                      validation_data=(X_validation, y_validation))

