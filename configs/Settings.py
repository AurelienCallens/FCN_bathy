#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global settings for Unet network"""

import tensorflow as tf
from src.utils import initialize_file_path

# Global settings
# MODE = "cpu"
MODE = "gpu"
CPU_CORES = 12

# 0) Input shape
DIR_NAME = "Data_ext"
IMG_SIZE = (512, 512)
N_CHANNELS = 3

# Data augmentation on train split
BRIGHT_R = [0.9, 1.1]
SHIFT_R = 0.1
ROT_R = 20
V_FLIP = True
H_FLIP = True

# 1) Unet Structure
ACTIV = "relu"
K_INIT = 'he_normal'
PRETRAINED_W = False
FILTERS = 16
NOISE_STD = 0.05
DROP_RATE = 0.2

# Pix2pix Structure
FILTERS_G = 16
FILTERS_D = 8

# 2) FCN training parameters
BATCH_SIZE = 6
EPOCHS = 100
LR = 0.002
OPTIMIZER = tf.keras.optimizers.Nadam()
OPTI_P = tf.keras.optimizers.Adam(0.0002, 0.5)

# 3) Callback parameters
# Decaying lr
DECAY_LR = True
INITIAL_LR = LR
FACTOR_DECAY = 0.8
N_EPOCHS_DECAY = 20

# Early stopping
PATIENCE = 20


# 4) File paths

train_input_img_paths, train_target_img_paths = initialize_file_path(DIR_NAME,
                                                                     'Train')
val_input_img_paths, val_target_img_paths = initialize_file_path(DIR_NAME,
                                                                 'Validation')
test_input_img_paths, test_target_img_paths = initialize_file_path(DIR_NAME,
                                                                   'Test')
