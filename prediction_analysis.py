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
mode = 'gpu'
start_tf_session(mode)

# keras seed fixing
seed(42)
# tensorflow seed fixing
tf.random.set_seed(42)



# 1) Load a model
Unet_model = UNet(size=img_size, bands=n_channels)

Trained_model = tf.keras.models.load_model('trained_models/cGAN_data_sup_1.1_01_31_noise_binary_loss',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[root_mean_squared_error,absolute_error, psnr, ssim, ms_ssim])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)

preds = Trained_model.predict(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)

########################################
# Display predictions with uncertainty #
########################################


pred_number = 50
item = 7
true_0 = test_gen.__getitem__(item)[1]
input_0 = test_gen.__getitem__(item)[0]
pred_0 = Trained_model.predict(test_gen.__getitem__(item)[0]).squeeze()
#pred_0 = Trained_model.predict(np.zeros((1, 512, 512, 3))
#Trained_model.summary()

for i in range(pred_number-1):
    pred_0 = np.dstack([pred_0 , Trained_model.predict(test_gen.__getitem__(item)[0]).squeeze()])

import matplotlib.pyplot as plt
_vmin, _vmax = np.min(true_0)-1, np.max(true_0)+1

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(6, 11)
gs.update(wspace=0.8, hspace=0.5)
ax1 = fig.add_subplot(gs[:3, 0:3])
ax1.imshow(np.uint8(input_0.squeeze()[:, :, 0]*255), cmap='gray')
ax2 = fig.add_subplot(gs[ 3:6, 0:3])
ax2.imshow(np.uint8(input_0.squeeze()[:, :, 1]*255), cmap='gray')
ax3 = fig.add_subplot(gs[ :3, 4:7])
im = ax3.imshow(true_0.squeeze().astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
ax4 = fig.add_subplot(gs[3:6, 4:7])
im = ax4.imshow(pred_0.mean(axis=2), cmap='jet', vmin=_vmin, vmax=_vmax)
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
ax5 = fig.add_subplot(gs[:3, 8:11])
#im = ax5.imshow(true_0.squeeze() - pred_0.mean(axis=2), cmap='bwr')
im = ax5.imshow(np.abs(true_0.squeeze() - pred_0.mean(axis=2)), cmap='inferno')
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
ax6 = fig.add_subplot(gs[3:6, 8:11])
im = ax6.imshow(pred_0.std(axis=2), cmap='inferno', vmin=0, vmax=(pred_0.std(axis=2)).max())
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
ax1.title.set_text('Input Snap')
ax2.title.set_text('Input Timex')
ax3.title.set_text('True bathy')
ax4.title.set_text('Pred. bathy')
ax5.title.set_text('Abs. Error')
ax6.title.set_text('Uncertainty')
plt.show()


########################################
#      Display activations map         #
########################################

Trained_model.summary()

model = tf.keras.Model(inputs=Trained_model.inputs,
                       outputs=Trained_model.layers[39].output)


features = model.predict(input_0)

fig = plt.figure(figsize=(20, 15))
for i in range(1, features.shape[3]+1):
    plt.subplot(6,6,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
plt.show()


im = plt.imshow(features.mean(axis=3).squeeze(), cmap='gray')
plt.colorbar(im, fraction=0.046, pad=0.04)


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


preds = Trained_model.predict(test_gen)

true = np.stack([ test_gen.__getitem__(i)[1] for i in range(test_gen.__len__())], axis=0)


def calculate_metrics(true, pred):
    return([root_mean_squared_error(true, pred).numpy(),
            absolute_error(true, pred).numpy(),
            psnr(true, pred).numpy()[0],
            ssim(true, pred).numpy()[0],
            ms_ssim(true, pred).numpy()[0]])

acc_array = np.array(list(map(lambda x, y: calculate_metrics(x, y), true, preds)))
acc_array = pd.DataFrame(acc_array, columns=('rmse', 'mae', 'pnsr', 'ssim', 'ms_ssim'))


acc_array['Date'] = list(map(lambda x: os.path.basename(x)[:-4], test_input_img_paths))
#acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H_%M_%S")
acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H:%M:%S")

tide_wave_cond = pd.read_csv('data_CNN/Data_processed/meta_df.csv')[['Date', 'bathy', 'Tide', 'Hs_m', 'Tp_m', 'Dir_m']]
tide_wave_cond['Date']= pd.to_datetime(tide_wave_cond['Date'], format="%Y-%m-%d %H:%M:%S")

acc_array = pd.merge(acc_array, tide_wave_cond, on='Date', how='inner').drop_duplicates(ignore_index=True)


acc_array.mean()

# Rmse depending on tide
acc_array['rip'] = 0
acc_array.loc[acc_array['bathy'].isin( ['2017-03-27', '2018-01-31']), 'rip'] = 1


sns.scatterplot(y='Tide', x='rmse', hue='rip', data=acc_array)
sns.scatterplot(y='Tide', x='rmse', hue='bathy', data=acc_array)

sns.scatterplot(y='Hs_m', x='rmse', hue='rip', data=acc_array)
sns.scatterplot(y='Hs_m', x='rmse', hue='bathy', data=acc_array)

sns.scatterplot(y='Tp_m', x='rmse', hue='rip', data=acc_array)
sns.scatterplot(y='Tp_m', x='rmse', hue='bathy', data=acc_array)

import plotly.express as px

fig = px.scatter_3d(acc_array, x='rmse', y='Tide', z='Hs_m')
fig.write_html("3D_scatter.html")

sns.heatmap(acc_array[acc_array['bathy'] == '2018-01-31'].corr(), annot=True, cmap='bwr')


acc_array['Acc'] = 0
acc_array.loc[acc_array['rmse'] < 0.4, 'Acc'] = 1

sns.boxplot(y = 'Tide', x = 'Acc', data = acc_array)

rip_bathy = acc_array[acc_array['rip'] == 1]
sns.boxplot(y = 'Tide', x = 'Acc', data = rip_bathy)
sns.boxplot(y = 'Tp_m', x = 'Acc', data = rip_bathy)


acc_array.loc[acc_array['Tide'] < 2.5].mean()
acc_array.mean()

acc_array.loc[:34, :].mean()

acc_array.loc[35:56, :].mean()

acc_array.loc[57:90, :].mean()

acc_array.loc[91:, :].mean()
