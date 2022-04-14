#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom generator"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Custom_gen(keras.utils.Sequence):
    """Custom generator class to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, bands, split):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.bands = bands
        if split == 'Train':
            self.augmentor = ImageDataGenerator(brightness_range=[0.9, 1.1],
                                                rescale=1/255)
        else:
            self.augmentor = ImageDataGenerator(rescale=1/255)

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (self.bands,), dtype=np.float16)
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            aug_img = next(self.augmentor.flow(np.expand_dims(img, axis=0))).squeeze()
            aug_img[:,:,2] = img[:,:,2]
            # (aug_img[:,:,2] == img[:,:,2]).all()
            # plt.imshow(aug_img)
            x[j] = aug_img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype=np.float16)
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y
