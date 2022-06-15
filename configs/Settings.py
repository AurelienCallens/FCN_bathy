#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global settings for Unet network"""

import tensorflow as tf
from ops.utils import initialize_file_path

# 0) Input shape
IMG_SIZE = (512, 512)
N_CHANNELS = 3

# Data augmentation on train split
BRIGHT_R = [0.9, 1.1]
SHIFT_R = 0.1

# 1) FCN Structure
ACTIV = "relu"
K_INIT = 'he_normal'
PRETRAINED_W = False
FILTERS = 16
NOISE_STD = 0.05
DROP_RATE = 0.2

# 2) FCN training parameters
BATCH_SIZE = 6
EPOCHS = 100
LR = 0.002
OPTIMIZERS = tf.keras.optimizers.Nadam()

# 3) Callback parameters
# Decaying lr
DECAY_LR = True
INITIAL_LR = LR
FACTOR_DECAY = 0.8
N_EPOCHS_DECAY = 20

# Early stopping
PATIENCE = 20


# 4) File paths
DIR_NAME = "Data_low_window"

train_input_img_paths, train_target_img_paths = initialize_file_path(DIR_NAME, 'Train')
val_input_img_paths, val_target_img_paths = initialize_file_path(DIR_NAME, 'Validation')
test_input_img_paths, test_target_img_paths = initialize_file_path(DIR_NAME, 'Test')
