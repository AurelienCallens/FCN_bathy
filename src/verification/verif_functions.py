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


def predict_novel_data(X_novel, Pred_novel):
    """Plot predictions of the network for novel data (no observed Y)

    Parameters
    ----------
    X_novel : numpy array
        X tensor
    Pred_novel : numpy array
        Prediction of the novel data

    Output
    ------
    Return plots of the predictions (Ŷ) with associated inputs.
    """
    _vmin, _vmax = np.max([np.min(pred_new)-1, -7]), np.max(pred_new)+1
    #_vmin, _vmax = np.min(pred_new)-1, np.max(pred_new)+1

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.8, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, :2], )
    ax1.imshow(np.uint8(X_novel.squeeze()[:, :, 0]*255), cmap='gray')
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(np.uint8(X_novel.squeeze()[:, :, 1]*255), cmap='gray')
    ax3 = fig.add_subplot(gs[0, 4:])
    ax3.imshow(np.uint8(X_novel.squeeze()[:, :, 2]*255), cmap='gray')
    ax4 = fig.add_subplot(gs[1, :3])
    im = ax4.imshow(gaussian_filter(Pred_novel.squeeze().astype('float32'), sigma=4), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax4)
    ax5 = fig.add_subplot(gs[1, 3:])
    im = ax5.contour(np.flipud(gaussian_filter(Pred_novel.squeeze().astype('float32'), sigma=4)),
                     cmap='jet', vmin=_vmin, vmax=_vmax, levels=10)
    plt.colorbar(im, ax=ax5)
    ax1.title.set_text('Mean RGB Snap')
    ax2.title.set_text('Mean RGB Timex')
    ax3.title.set_text('Env. Cond.')
    ax4.title.set_text('Pred. bathy')
    ax5.title.set_text('Pred. bathy')
    plt.show()


def plot_pred_uncertainty(item, test_gen, avg_pred, std_pred, err_pred):
    """Plot predictions of the network with error and uncertainty for an
    indicated image

    Parameters
    ----------
    item : int
        Index of the image to be plotted 
    test_gen :
        Image generator created with the CustomGenerator class for the test
    avg_pred : numpy array
        Averaged predictions for test data
    std_pred : numpy array
        Standard errors (uncertainty) associated to the predictions for test data
    err_pred : numpy array
        Absolute errors associated to the predictions for test data
    Output
    ------
    Return plots of the predictions (Ŷ) with associated inputs.
    """
    snap = test_gen.__getitem__(item)[0].squeeze()[:, :, 0]*255
    timex = test_gen.__getitem__(item)[0].squeeze()[:, :, 1]*255
    true = test_gen.__getitem__(item)[1].squeeze().astype('float32')

    _vmin, _vmax = np.min(true)-1, np.max(true)+1

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(6, 11)
    gs.update(wspace=0.8, hspace=0.5)
    ax1 = fig.add_subplot(gs[:3, 0:3])
    ax1.imshow(np.uint8(snap), cmap='gray')
    ax2 = fig.add_subplot(gs[ 3:6, 0:3])
    ax2.imshow(np.uint8(timex), cmap='gray')
    ax3 = fig.add_subplot(gs[ :3, 4:7])
    im = ax3.imshow(true.astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax4 = fig.add_subplot(gs[3:6, 4:7])
    im = ax4.imshow(avg_pred[item], cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    ax5 = fig.add_subplot(gs[:3, 8:11])
    im = ax5.imshow(err_pred[item], cmap='inferno')
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    ax6 = fig.add_subplot(gs[3:6, 8:11])
    im = ax6.imshow(std_pred[item]*2, cmap='inferno', vmin=0, vmax=std_pred[item].max()*2)
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    ax1.title.set_text('Input Snap')
    ax2.title.set_text('Input Timex')
    ax3.title.set_text('True bathy')
    ax4.title.set_text('Pred. bathy')
    ax5.title.set_text('Abs. Error')
    ax6.title.set_text('Uncertainty')
    plt.show()


# Make averaged predictions

def averaged_pred(test_gen, Trained_model, rep):
    """Function to return averaged predictions, associated standard deviations 
    and errors

    Parameters
    ----------
    test_gen :
        Image generator created with the CustomGenerator class for the test
        split
    Trained_model :
        Trained model
    rep : int
        Number of predictions to average

    Output
    ------
    List with average pred, standard deviation and errors
    """
    avg_pred = []
    std_pred = []
    err_pred = []
    for j in range(test_gen.__len__()):
        true_0 = test_gen.__getitem__(j)[1]
        input_0 = test_gen.__getitem__(j)[0]
        pred_0 = Trained_model.predict(test_gen.__getitem__(j)[0]).squeeze()

        for i in range(rep-1):
            pred_0 = np.dstack([pred_0 , Trained_model.predict(test_gen.__getitem__(j)[0]).squeeze()])

        avg_pred.append(pred_0.mean(axis=2))
        std_pred.append(pred_0.std(axis=2))
        err_pred.append(np.abs(true_0.squeeze() - pred_0.mean(axis=2)))
    return [avg_pred, std_pred, err_pred]

# Apply black patch to an images to perform pertubation analysis

def apply_black_patch(image, top_left_x, top_left_y, patch_size):
    """Apply black patch to an images to perform pertubation analysis

    Parameters
    ----------
    image : array
        Image
    top_left_s : int
        X oordinate of the top left corner of the patch
    top_left_y : int
        Y Coordinate of the top left corner of the patch
     patch_size : int
        Size in pixel of the patch

    Output
    ------
    Image with a black patch at the indicated location
    """
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:(top_left_y + patch_size), top_left_x:(top_left_x + patch_size), :] = 0

    return patched_image

