#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diverse functions to perform verifications for the network. They include:
    + functions to verify the absence of Nan data in the generators
    + function to plot the generator output to ensure for correct data
    augmentation procedures
    + function to plot history of network training
    + function to plot predictions of the network

Usage:
    from src.verification import *

Author:
    Aurelien Callens - 18/08/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# Checking no Nan in images


def check_nan_generator(img_generator):
    """Verify the absence of Nan data in the generators

    Parameters
    ----------
    img_generator :
        Image generator created with the CustomGenerator class

    Output
    ------
    Print if there is at least one Nan in the pixels of each images in the
    data flow of the generator
    """
    img_generator
    res_input = []
    res_target = []
    for i in range(img_generator.__len__()):
        img = img_generator.__getitem__(i)
        res_input.append(np.isnan(img[0]).any())
        res_target.append(np.isnan(img[1]).any())
    if sum(res_input) != 0 or sum(res_target) != 0:
        print("Nan in pixel values!")
    else:
        print("No Nan values!")


def check_nan_generator_unique(img_generator):
    """Check wether X or Y have Nan data in the generator flow

    Parameters
    ----------
    img_generator :
        Image generator created with the CustomGenerator class

    Output
    ------
    Print if there is at least one Nan in the pixels of each images in X or Y
    in the data flow of the generator
    """
    img_generator
    for i in range(img_generator.__len__()):
        img = img_generator.__getitem__(i)
        print("X:", np.isnan(img[0]).any())
        print("y:", np.isnan(img[1]).any())


# Plot output of train generator


def plot_output_generator(train_gen, n_img):
    """Plot the generator output to ensure for correct data
    augmentation procedures

    Parameters
    ----------
    train_gen :
        Image generator created with the CustomGenerator class for the
                train split
    n_img : int
        number of images to be plotted

    Output
    ------
    Return plots of n_img from the train generator
    """

    for i in range(n_img):
        test = train_gen.__getitem__(i)
        plt.subplot(221)
        plt.imshow(test[0][0][:, :, 0].astype(np.float32), cmap='gray')
        plt.subplot(222)
        plt.imshow(test[0][0][:, :, 1].astype(np.float32), cmap='gray')
        plt.subplot(223)
        plt.imshow(test[0][0][:, :, 2].astype(np.float32))
        plt.subplot(224)
        plt.imshow(test[1][0].squeeze().astype(np.float32))
        plt.show()


# Plot history


def plot_history(history):
    """Plot the history of network training

    Parameters
    ----------
    history :
        History of the trained network

    Output
    ------
    Return plots of the loss and metric history for the deep learning model
    """

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# Plot predictions


def plot_predictions(test_generator, predictions, every_n):
    """Plot predictions of the network

    Parameters
    ----------
    test_generator :
        Image generator created with the CustomGenerator class for the test
        split
    predictions : list
        List containing all the predictions made by the models on the
        test set
    every_n : int
        To not plot all the images. Plot one image every "n" images.

    Output
    ------
    Return plots of the predictions (Ŷ) with associated inputs and true Y.
    """

    for i in np.arange(0, predictions.shape[0], every_n):
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