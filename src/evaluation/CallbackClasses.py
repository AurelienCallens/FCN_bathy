#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two classes for customized keras callbacks

Usage:
    from src.evaluation.CallbackClasses import TimingCallback, StepDecay

Author:
    https://www.autoscripts.net/how-to-decrease-the-learning-rate-every-10-epochs-by-a-factor-of-0-9/#how-to-decrease-the-learning-rate-every-10-epochs-by-a-factor-of-09
"""

import tensorflow
import numpy as np
from timeit import default_timer as timer


class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.1, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)


# Callback to record training time
class TimingCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
