#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification functions for the network"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

## Checking no Nan in images
def check_nan_generator(model, split):

    img_generator = model.data_generator(split)

    for i in range(img_generator.__len__()):
        img = img_generator.__getitem__(i)
        print(np.isnan(img[0]).any())
        print(np.isnan(img[1]).any())


## Plot output of train generator
def plot_output_generator(model, n_img):
    train_gen = model.data_generator('Train')
    for i in range(n_img):
        test = train_gen.__getitem__(i)
        plt.subplot(221)
        plt.imshow(test[0][0][:,:,0].astype(np.float32), cmap='gray')
        plt.subplot(222)
        plt.imshow(test[0][0][:,:,1].astype(np.float32), cmap='gray')
        plt.subplot(223)
        plt.imshow(test[0][0][:,:,2].astype(np.float32))
        plt.subplot(224)
        plt.imshow(test[1][0].squeeze().astype(np.float32))
        plt.show()

## Plot history
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

## Plot predictions
def plot_predictions(test_generator, predictions, every_n):
    
    for i in np.arange(0, predictions.shape[0], 1):
        cond_img = test_generator.__getitem__(i)[0]
        true_img = test_generator.__getitem__(i)[1]
        pred_img = predictions[i, :, :, :]
        _vmin, _vmax = np.min(true_img)-1, np.max(true_img)+1
        
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 6)
        gs.update(wspace=0.8, hspace=0.5)
        ax1 = fig.add_subplot(gs[0, :2], )
        ax1.imshow(np.uint8(cond_img.squeeze()[:, :, 0]*255), cmap='gray')
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.imshow(np.uint8(cond_img.squeeze()[:, :, 1]*255), cmap='gray')
        ax3 = fig.add_subplot(gs[0, 4:])
        ax3.imshow(np.uint8(cond_img.squeeze()[:, :, 2]*255), cmap='gray')
        ax4 = fig.add_subplot(gs[1, 0:3])
        im = ax4.imshow(true_img.squeeze().astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
        plt.colorbar(im, ax=ax4)
        ax5 = fig.add_subplot(gs[1, 3:])
        im = ax5.imshow(gaussian_filter(pred_img.squeeze().astype('float32'), sigma=6), cmap='jet', vmin=_vmin, vmax=_vmax)
        plt.colorbar(im, ax=ax5)
        ax1.title.set_text('Mean RGB Snap')
        ax2.title.set_text('Mean RGB Timex')
        ax3.title.set_text('Env. Cond.')
        ax4.title.set_text('True bathy')
        ax5.title.set_text('Pred. bathy')
        plt.show()