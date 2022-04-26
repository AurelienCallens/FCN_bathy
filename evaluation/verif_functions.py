#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification functions for the network"""

import numpy as np
import matplotlib.pyplot as plt


## Checking no Nan in images
def check_nan_generator(model):
    train_generator, val_generator = model.data_generator('Train')
    for i in range(train_generator.__len__()):
        img = train_generator.__getitem__(i)
        #plt.imshow( img[0][0].astype(np.float32))
        #plt.show()
        print(np.isnan(img[0]).any())
        print(np.isnan(img[1]).any())

## Plot history
def plot_history(history, name_plot):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(name_plot)

