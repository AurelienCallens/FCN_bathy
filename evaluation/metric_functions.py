#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Different metric functions taken from Collins et al. 2021 (https://github.com/collins-frf/Celerity_Net/blob/master/losses.py)"""

import tensorflow as tf
import numpy as np

def pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error)
    loss.__name__ = "pixel_error"

def absolute_error(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    return tf.reduce_mean(abs_error)
    loss.__name__ = "absolute_error"

def pred_max(target, output):
    return tf.reduce_max(output)
    loss.__name__ = "pred_max"

def pred_min(target, output):
    return tf.reduce_min(output)
    loss.__name__ = "pred_min"