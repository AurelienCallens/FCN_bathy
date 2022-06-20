#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Different metric functions taken from Collins et al. 2021
(https://github.com/collins-frf/Celerity_Net/blob/master/losses.py)"""

import tensorflow as tf

# Common metrics for regression

# Mean absolute error

def absolute_error(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    return tf.reduce_mean(abs_error)
    loss.__name__ = "mae"

# Root mean squared error

def root_mean_squared_error(y_true, y_pred):
    error = y_pred - y_true
    squared_error = tf.reduce_mean(error**2)
    return tf.math.sqrt(squared_error)
    loss.__name__ = "rmse"

# Maximun predicted value

def pred_max(target, output):
    return tf.reduce_max(output)
    loss.__name__ = "pred_max"

# Minimun predicted value

def pred_min(target, output):
    return tf.reduce_min(output)
    loss.__name__ = "pred_min"

# Specific image similarity metrics

# Peak Signal-to-Noise Ratio
# https://ieeexplore.ieee.org/abstract/document/1284395

def psnr(y_true, y_pred):
    max_val = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val)
    loss.__name__ = "psnr"

# Structural Similarity Index
# https://ieeexplore.ieee.org/abstract/document/1284395

def ssim(y_true,y_pred):
    max_val = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val)
    loss.__name__ = "ssim"

# Multi-scale Structural Similarity Index
# https://ieeexplore.ieee.org/abstract/document/1292216

def ms_ssim(y_true,y_pred):
    max_val = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.ssim_multiscale(y_true, y_pred, max_val)
    loss.__name__ = "ms_ssim"

