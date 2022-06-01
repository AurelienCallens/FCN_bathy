#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global settings for Unet network"""

import tensorflow as tf
from ops.utils import initialize_file_path
# 0) Input shape
img_size = (512, 512)
n_channels = 3

# Data augmentation on train split
brightness_r = [0.9, 1.1]
shift_range = 0.3

# 1) FCN Structure
activ = "relu"
k_init = 'he_normal'
pretrained_weights = False
filters = 16
noise_std = 0.05
drop_rate = 0.2

# 2) FCN training parameters
batch_size = 6
n_epochs = 10
lr = 0.002
optimizer = tf.keras.optimizers.Nadam()
#optimizer = tf.keras.optimizers.SGD(momentum=0.9)

# 3) Callback parameters
# Decaying lr
decaying_lr = True
initial_lr = lr
factor_decay = 0.8
nb_epoch_decay = 20
# Early stopping
epoch_patience = 20


# 4) File paths
dir_name = "Data_new_sup_0.9"

train_input_img_paths, train_target_img_paths = initialize_file_path(dir_name, 'Train')
val_input_img_paths, val_target_img_paths = initialize_file_path(dir_name, 'Validation')
test_input_img_paths, test_target_img_paths = initialize_file_path(dir_name, 'Test')
