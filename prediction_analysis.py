#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:58:37 2022

@author: aurelien
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import seed
from configs.Settings import *
from model.UnetModel import UNet
from evaluation.verif_functions import *
from evaluation.metric_functions import *
from executor.tf_init import start_tf_session
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import Model
from scipy.ndimage import gaussian_filter

# 0) Initialize session
#mode = 'cpu'
#mode = 'gpu'
#start_tf_session(mode)

# keras seed fixing
seed(42)
# tensorflow seed fixing
tf.random.set_seed(42)



# 1) Load a model
Unet_model = UNet(size=img_size, bands=n_channels)

Trained_model = tf.keras.models.load_model('trained_models/cGAN_data_sup_1.1_noise_binary_loss',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[root_mean_squared_error,absolute_error, psnr, ssim, ms_ssim])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)

train_gen = Unet_model.data_generator('Train')
Trained_model.evaluate(train_gen)

preds = Trained_model.predict(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=5)

########################################
# Display predictions with uncertainty #
########################################


def averaged_pred(test_gen, Trained_model, rep):
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

avg_pred, std_pred, err_pred = averaged_pred(test_gen, Trained_model, 20)


item = 20
snap = test_gen.__getitem__(item)[0].squeeze()[:, :, 0]*255
timex = test_gen.__getitem__(item)[0].squeeze()[:, :, 1]*255
true = test_gen.__getitem__(item)[1].squeeze().astype('float32')

import matplotlib.pyplot as plt
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
#im = ax5.imshow(true_0.squeeze() - pred_0.mean(axis=2), cmap='bwr')
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


std_vec = np.array(std_pred)[:20, :, :].flatten()
err_vec = np.array(err_pred)[:20, :, :].flatten()

plt.scatter(x=std_vec, y=err_vec)
plt.xlabel("Uncertainty")
plt.ylabel("Absolute error in meters")
plt.title("Absolute error vs uncertainty for the first 20 images")

########################################
#      Display activations map         #
########################################

Trained_model.summary()

model = tf.keras.Model(inputs=Trained_model.inputs,
                       outputs=Trained_model.layers[48].output)

input_0 = test_gen.__getitem__(item)[0]
features = model.predict(input_0)

fig = plt.figure(figsize=(20, 15))
for i in range(1, features.shape[3]+1):
    plt.subplot(4,4,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
plt.show()


im = plt.imshow(features.mean(axis=3).squeeze(), cmap='gray')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(6, 9)
gs.update(wspace=0.8, hspace=0.5)
ax1 = fig.add_subplot(gs[:6, 0:3])
ax1.imshow(np.uint8(snap), cmap='gray')
ax2 = fig.add_subplot(gs[:6 , 3:6])
ax2.imshow(np.uint8(timex), cmap='gray')
ax2 = fig.add_subplot(gs[:6 , 6:])
ax2.imshow(np.uint8(test_gen.__getitem__(item)[0].squeeze()[:, :, 2]*255), cmap='gray')

########################################
#           Occlusion map              #
########################################


img = input_0[0]

def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:(top_left_y + patch_size), top_left_x:(top_left_x + patch_size), :] = 0

    return patched_image

pred_0 = Trained_model.predict(input_0)
PATCH_SIZE = 32
sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

# Iterate the patch over the image
for top_left_x in range(0, img.shape[0], PATCH_SIZE):
    for top_left_y in range(0, img.shape[1], PATCH_SIZE):
        patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)
        predicted_img = Trained_model.predict(np.array([patched_image]))[0]

        confidence = float(absolute_error(true_0, predicted_img)/absolute_error(true_0, pred_0))

        # Save confidence for this specific patched image in map
        sensitivity_map[
            top_left_y:top_left_y + PATCH_SIZE,
            top_left_x:top_left_x + PATCH_SIZE,
        ] = confidence

im = plt.imshow(gaussian_filter(sensitivity_map, 5), cmap='jet')
plt.colorbar(im,  fraction=0.046, pad=0.04)


########################################
# Accuracy vs environmental conditions #
########################################
preds = []
preds_std = []
for i in range(test_gen.__len__()):
    pred_number = 50
    item = i
    true_0 = test_gen.__getitem__(item)[1]
    input_0 = test_gen.__getitem__(item)[0]
    pred_0 = Trained_model.predict(test_gen.__getitem__(item)[0]).squeeze()

    for i in range(pred_number-1):
        pred_0 = np.dstack([pred_0 , Trained_model.predict(test_gen.__getitem__(item)[0]).squeeze()])
    preds.append(pred_0.mean(axis=2))
    preds_std.append(pred_0.std(axis=2))


#preds = Trained_model.predict(test_gen)

true = np.stack([ test_gen.__getitem__(i)[1] for i in range(test_gen.__len__())], axis=0)


def calculate_metrics(true, pred):
    return([root_mean_squared_error(true, pred).numpy(),
            absolute_error(true, pred).numpy(),
            psnr(true, pred).numpy()[0],
            ssim(true, pred).numpy()[0],
            ms_ssim(true, pred).numpy()[0]])

acc_array = np.array(list(map(lambda x, y: calculate_metrics(x, np.expand_dims(y,2)), true, preds)))
acc_array = pd.DataFrame(acc_array, columns=('rmse', 'mae', 'pnsr', 'ssim', 'ms_ssim'))
acc_array['std'] = list(map(np.mean, preds_std))

acc_array['Date'] = list(map(lambda x: os.path.basename(x)[:-4], test_input_img_paths))
#acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H_%M_%S")
acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H_%M_%S")

tide_wave_cond = pd.read_csv('data_CNN/Data_processed/meta_df.csv')[['Date', 'bathy', 'Tide', 'Hs_m', 'Tp_m', 'Dir_m']]
tide_wave_cond['Date']= pd.to_datetime(tide_wave_cond['Date'], format="%Y-%m-%d %H:%M:%S")

acc_array = pd.merge(acc_array, tide_wave_cond, on='Date', how='inner').drop_duplicates(ignore_index=True)


acc_array.mean()

# Rmse depending on tide
acc_array['rip'] = 0
acc_array.loc[acc_array['bathy'].isin( ['2017-03-27', '2018-01-31']), 'rip'] = 1

acc_array.to_csv('Accuracy_test_set.csv')
