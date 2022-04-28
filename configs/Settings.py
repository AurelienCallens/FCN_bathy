#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global settings for Unet network"""

import tensorflow as tf
from ops.utils import sorted_list_path
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
filters = 4
noise_std = 0.05
drop_rate = 0.2

# 2) FCN training parameters
batch_size = 6
n_epochs = 500
lr = 0.003
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
dir_name = "Data_sup_1.1"
input_dir_train = "./data_CNN/" + dir_name + "/Train/Input/"
target_dir_train = "./data_CNN/" + dir_name + "/Train/Target/"
input_dir_val = "./data_CNN/" + dir_name + "/Validation/Input/"
target_dir_val = "./data_CNN/" + dir_name + "/Validation/Target/"
input_dir_test = "./data_CNN/" + dir_name + "/Test/Input/"
target_dir_test = "./data_CNN/" + dir_name + "/Test/Target/"


train_input_img_paths = sorted_list_path(input_dir_train)
train_target_img_paths = sorted_list_path(target_dir_train)
val_input_img_paths = sorted_list_path(input_dir_val)
val_target_img_paths = sorted_list_path(target_dir_val)
test_input_img_paths = sorted_list_path(input_dir_test)
test_target_img_paths = sorted_list_path(target_dir_test)


