#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom generator"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configs.Settings import *


class CustomGenerator(keras.utils.Sequence):
    """Custom generator class to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, bands, split):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.bands = bands
        self.split = split

        if split in ['Validation', 'Test', 'Train_no_aug']:
            self.augmentor = ImageDataGenerator(rescale=1/255)

        elif split == 'Train':
            self.augmentor = ImageDataGenerator(brightness_range=BRIGHT_R,
                                                rescale=1/255)
            self.shift_augmentor = ImageDataGenerator(height_shift_range=SHIFT_R,
                                                      width_shift_range=SHIFT_R,
                                                      horizontal_flip=H_FLIP,
                                                      vertical_flip=V_FLIP,
                                                      rotation_range=ROT_R)
        else:
            print('Wrong split type!')

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (self.bands,),
                     dtype=np.float16)
        y = np.zeros((self.batch_size,) + self.img_size + (1,),
                     dtype=np.float16)

        assert(len(batch_input_img_paths) == len(batch_target_img_paths))

        for j in range(len(batch_input_img_paths)):
            X_img = np.load(batch_input_img_paths[j])
            Y_img = np.load(batch_target_img_paths[j])

            if self.split == 'Train':
                cond = np.copy(X_img[:, :, 2])
                X_img[:, :, 2] = Y_img

                # Same shift generator on 1st and 2nd channels of X and on Y
                aug_img = next(
                    self.shift_augmentor.flow((np.expand_dims(X_img, axis=0)),
                                              shuffle=False))
                Y_img[:] = aug_img[:, :, :, 2]

                # Generator for brightness + scaling on 1st, 2nd channels of X
                aug_img_2 = next(self.augmentor.flow((aug_img)))
                final_X = aug_img_2.squeeze()
                final_X[:, :, 2] = cond

            else:
                final_X = next(self.augmentor.flow((np.expand_dims(X_img,
                                                                   axis=0)),
                                                   shuffle=False))
                final_X = final_X.squeeze()
                final_X[:, :, 2] = X_img[:, :, 2]

            x[j] = final_X
            y[j] = np.expand_dims(Y_img.squeeze(), 2)

        return x, y


