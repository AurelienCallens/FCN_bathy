#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:43:32 2022

@author: aurelien
"""

import numpy as np
from numpy.random import seed
import tensorflow as tf
from configs.Settings import *
import matplotlib.pyplot as plt
from model.Unet_model import UNet
from scipy.ndimage import gaussian_filter
from executor.tf_init import start_tf_sesssion
from evaluation.Verif_functions import plot_history, check_nan_generator
from evaluation.Metric_functions import pixel_error, absolute_error, pred_min, pred_max
import matplotlib.gridspec as gridspec


# 0) Initialize session
mode = 'cpu'
#mode = 'gpu'
start_tf_sesssion(mode)

seed(42)# keras seed fixing 
tf.random.set_seed(42)# tensorflow seed fixing

# 1) Test one model

Unet_model = UNet(size=img_size, bands=n_channels)

# Check FCN structure
model = Unet_model.build()
model.summary()

# Check if Nan values in train gen
# check_nan_generator(Unet_model)

# test_gen = Unet_model.data_generator('Test')
# test = test_gen.__getitem__(0)

# train_gen, val_gen = Unet_model.data_generator('Train')
# test = train_gen.__getitem__(0)

# plt.imshow(test[0][0][:,:,0].astype(np.float32), cmap='gray')
# plt.imshow(test[0][0][:,:,1].astype(np.float32), cmap='gray')
# plt.imshow(test[0][0][:,:,2].astype(np.float32))
# plt.imshow(test[1][0].squeeze().astype(np.float32))

# Train the model
Trained_model = Unet_model.train()

# Verification of the training loss
name = 'trained_models/Model_1'
plot_history(Trained_model[1], name + '_bathy.png')

preds = Unet_model.predict()
train_gen, val_gen = Unet_model.data_generator('Train')
test_gen = Unet_model.data_generator('Test')
Trained_model[0].evaluate(test_gen)

for i in np.arange(0, preds.shape[0], 3):
    true_img = test_gen.__getitem__(i)[1]
    pred_img = preds[i, :, :, :]
    # _vmin, _vmax = np.min([np.min(true_img), np.min(pred_img)]), np.max([np.max(true_img), np.max(pred_img)])
    _vmin, _vmax = np.min(true_img)-1, np.max(true_img)+1
    plt.subplot(1,2,1)
    plt.imshow(true_img.squeeze().astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.subplot(1,2,2)
    plt.imshow(gaussian_filter(pred_img.squeeze().astype('float32'), sigma=8), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.show()

tf.keras.models.save_model(Trained_model[0], name)

# 2) Test several models
batch_size = np.arange(4, 24, 4)
filter_size = np.arange(8, 49, 8)
combs = [(i, j) for i in batch_size for j in filter_size]

for comb in combs:
   batch_size = comb[0]
   filters = comb[1]
   fcn_model = UNet(size=img_size, bands=n_channels, model_saved=None)
   try:
       temp_model = fcn_model.train()
       test_gen = fcn_model.data_generator('Test')
       acc = np.round(temp_model[0].evaluate(test_gen), 3)
       name = 'FCN_acc_' + str(acc) + '_b_' + str(batch_size) + '_f_' + str(filters) + '_lr0.003_decay_0.8_40'
       temp_model[0].save(name)
       plot_history(temp_model[1], name + '.png')
   except MemoryError:
       pass


# 3) Load a model
Trained_model = tf.keras.models.load_model('trained_models/CNN_allbathy_newgen_batch6_filter16_lr0.001_decay_0.8_every40_activ_relu_bathy',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[absolute_error, pred_min, pred_max])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)
preds = Trained_model.predict(test_gen)

for i in np.arange(0, preds.shape[0], 1):
    cond_img = test_gen.__getitem__(i)[0]
    true_img = test_gen.__getitem__(i)[1]
    pred_img = preds[i, :, :, :]
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


