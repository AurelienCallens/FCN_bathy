#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:43:32 2022

@author: aurelien
"""

import numpy as np
from configs.Settings import *
import matplotlib.pyplot as plt
from model.Unet_model import UNet
from scipy.ndimage import gaussian_filter
from executor.tf_init import start_tf_sesssion
from evaluation.Verif_functions import plot_history, check_nan_generator

# 0) Initialize session
mode = 'gpu'
start_tf_sesssion(mode)

# 1) Test one model

Unet_model = UNet(size=img_size, bands=n_channels, model_saved=None)

# Check FCN structure
model = Unet_model.build()
model.summary()

# Check if Nan values in train gen
# check_nan_generator(Unet_model)

# Train the model
Trained_model = Unet_model.train()

# Verification of the training loss
name = 'CNN_batch' + str(batch_size) + '_filter' + str(filters) + '_lr' + str(lr) + '_decay_' + str(factor_decay) + '_every' + str(nb_epoch_decay) + '_activ_' + str(activ)
plot_history(Trained_model[1], name + '.png')

preds = Unet_model.predict()
test_gen = Unet_model.data_generator('Test')
Trained_model[0].evaluate(test_gen)

for i in np.arange(0, preds.shape[0], 3):
    true_img = test_gen.__getitem__(i)[1]
    pred_img = preds[i, :, :, :]
    _vmin, _vmax = np.min([np.min(true_img), np.min(pred_img)]), np.max([np.max(true_img), np.max(pred_img)])
    plt.subplot(1,2,1)
    plt.imshow(true_img.squeeze().astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.subplot(1,2,2)
    plt.imshow(gaussian_filter(pred_img.squeeze().astype('float32'), sigma=8), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.show()

Trained_model[0].save(name)


# 3) Test several models
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







