#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global settings for Unet network"""

import os
import tensorflow as tf

# 0) Input shape

img_size = (512, 512)
n_channels = 3

# 1) FCN Structure

activ = "sigmoid"
k_init = 'he_normal'
pretrained_weights = False
filters = 4
noise_std = 0.05

# 2) FCN training parameters
batch_size = 12
n_epochs = 1
lr = 0.001
optimizer = tf.keras.optimizers.Nadam()

# 3) Callback parameters
# Decaying lr
decaying_lr = True
initial_lr = lr
factor_decay = 0.8
nb_epoch_decay = 40

# Early stopping
epoch_patience = 20


# 4) File paths
input_dir_train = "data_CNN/Data/Train/Input/"
target_dir_train = "data_CNN/Data/Train/Target/"
input_dir_test = "data_CNN/Data/Test/Input/"
target_dir_test = "data_CNN/Data/Test/Target/"

train_input_img_paths = sorted(
    [
        os.path.join(input_dir_train, fname)
        for fname in os.listdir(input_dir_train)
        if fname.endswith(".npy")
    ]
)
train_target_img_paths = sorted(
    [
        os.path.join(target_dir_train, fname)
        for fname in os.listdir(target_dir_train)
        if fname.endswith(".npy")
    ]
)

test_input_img_paths = sorted(
    [
        os.path.join(input_dir_test, fname)
        for fname in os.listdir(input_dir_test)
        if fname.endswith(".npy")
    ]
)
test_target_img_paths = sorted(
    [
        os.path.join(target_dir_test, fname)
        for fname in os.listdir(target_dir_test)
        if fname.endswith(".npy")
    ]
)


