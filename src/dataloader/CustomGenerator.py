#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom generator"""

import numpy as np
from tensorflow.image import resize
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CustomGenerator(keras.utils.Sequence):
    """Custom generator class to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, params, input_img_paths, target_img_paths, split):

        self.BATCH_SIZE = batch_size
        self.IMG_SIZE = eval(params['Input']['IMG_SIZE'])
        self.BANDS = params['Input']['N_CHANNELS']
        self.BRIGHT_R = eval(params['Data_aug']['BRIGHT_R'])
        self.SHIFT_R = params['Data_aug']['SHIFT_R']
        self.H_FLIP = params['Data_aug']['H_FLIP']
        self.V_FLIP = params['Data_aug']['V_FLIP']
        self.ROT_R = params['Data_aug']['ROT_R']

        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.split = split


        if split in ['Validation', 'Test', 'Train_no_aug']:
            self.augmentor = ImageDataGenerator(rescale=1/255)

        elif split == 'Train':
            self.augmentor = ImageDataGenerator(brightness_range=self.BRIGHT_R,
                                                rescale=1/255)
            self.shift_augmentor = ImageDataGenerator(height_shift_range=self.SHIFT_R,
                                                      width_shift_range=self.SHIFT_R,
                                                      horizontal_flip=self.H_FLIP,
                                                      vertical_flip=self.V_FLIP,
                                                      rotation_range=self.ROT_R)
        else:
            print('Wrong split type!')

    def __len__(self):
        return len(self.target_img_paths) // self.BATCH_SIZE

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        i = idx * self.BATCH_SIZE
        batch_input_img_paths = self.input_img_paths[i: i + self.BATCH_SIZE]
        batch_target_img_paths = self.target_img_paths[i: i + self.BATCH_SIZE]

        x = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (self.BANDS,),
                     dtype=np.float16)
        y = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (1,),
                     dtype=np.float16)

        assert(len(batch_input_img_paths) == len(batch_target_img_paths))

        for j in range(len(batch_input_img_paths)):
            X_img = np.load(batch_input_img_paths[j])
            Y_img = np.load(batch_target_img_paths[j])

            if(not self.IMG_SIZE[0] == 512):
                X_img = np.array(resize(X_img, self.IMG_SIZE), dtype=np.float16)
                Y_img = np.array(resize(np.expand_dims(Y_img, axis=2), self.IMG_SIZE), dtype=np.float16).squeeze()

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



