#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification functions for the network

Usage:


Author:
    Aurelien Callens - 05/05/2022
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from src.models.UnetModel import UNet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from src.Param import Param
from src.executor.tf_init import start_tf_session
from src.utils import initialize_file_path
from src.evaluation.metric_functions import *
from src.verification.verif_functions import plot_predictions, averaged_pred, apply_black_patch
from src.utils import sorted_list_path

# 0) Initialize session
# mode = 'cpu'
mode = 'cpu'
start_tf_session(mode)

# keras seed fixing
seed(42)
# tensorflow seed fixing
tf.random.set_seed(42)

case = 32
res_csv = pd.read_csv('trained_models/Results_test.csv')

params = Param('./configs/' + res_csv['Param_file'][case]).load()

train_input, train_target = initialize_file_path(params['Input']['DIR_NAME'], 'Train')
val_input, val_target = initialize_file_path(params['Input']['DIR_NAME'], 'Validation')
test_input, test_target = initialize_file_path(params['Input']['DIR_NAME'], 'Test')
test_test = sorted_list_path("./data_CNN/Test_data/Test/Input/")

# 1) Load a model
Unet_model = UNet(params)

Trained_model = tf.keras.models.load_model(res_csv['Name'][case],
                                           custom_objects={'absolute_error': absolute_error,
                                                           'pred_min': pred_min,
                                                           'pred_max': pred_max},
                                           compile=False)

Trained_model.compile(optimizer=Adam(params['Train']['LR'], 0.5), loss='mse', metrics=[root_mean_squared_error,absolute_error, psnr, ssim, ms_ssim])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)

train_gen = Unet_model.data_generator('Train_no_aug')
Trained_model.evaluate(train_gen)

############### Test on unseen data 

X_img = np.load(test_test[0])
X_img.shape
np.expand_dims(X_img, axis=0)
pred_new = Trained_model.predict(np.expand_dims(X_img, axis=0))

plt.imshow(pred_new.squeeze())

_vmin, _vmax = np.min(pred_new)-1, np.max(pred_new)+1

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.8, hspace=0.5)
ax1 = fig.add_subplot(gs[0, :2], )
ax1.imshow(np.uint8(X_img.squeeze()[:, :, 0]*255), cmap='gray')
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.imshow(np.uint8(X_img.squeeze()[:, :, 1]*255), cmap='gray')
ax3 = fig.add_subplot(gs[0, 4:])
ax3.imshow(np.uint8(X_img.squeeze()[:, :, 2]*255), cmap='gray')
ax4 = fig.add_subplot(gs[1, :3])
im = ax4.imshow(gaussian_filter(pred_new.squeeze().astype('float32'), sigma=4), cmap='jet', vmin=_vmin, vmax=_vmax)
plt.colorbar(im, ax=ax4)
ax5 = fig.add_subplot(gs[1, 3:])
xs, ys = np.arange(0,512), np.arange(0,512)
xgrid, ygrid = np.meshgrid(xs, ys)
xdata, ydata = np.ravel(xgrid), np.ravel(ygrid)
#im = ax5.tricontour(xdata, ydata, np.ravel(gaussian_filter(pred_new.squeeze().astype('float32'), sigma=5)),
#                cmap='jet', vmin=_vmin, vmax=_vmax)

im = ax5.contour(np.flipud(gaussian_filter(pred_new.squeeze().astype('float32'), sigma=4)),
                 cmap='jet', vmin=_vmin, vmax=_vmax, levels=10)
plt.colorbar(im, ax=ax5)
ax1.title.set_text('Mean RGB Snap')
ax2.title.set_text('Mean RGB Timex')
ax3.title.set_text('Env. Cond.')
ax4.title.set_text('Pred. bathy')
ax5.title.set_text('Pred. bathy')
plt.show()

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)

########################################
# Display predictions with uncertainty #
########################################


avg_pred, std_pred, err_pred = averaged_pred(test_gen, Trained_model, 20)


item = 20
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
true_0 = test_gen.__getitem__(item)[1]

pred_0 = Trained_model.predict(input_0)
PATCH_SIZE = 32
sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

# Iterate the patch over the image
for top_left_x in range(0, img.shape[0], PATCH_SIZE):
    for top_left_y in range(0, img.shape[1], PATCH_SIZE):
        patched_image = apply_black_patch(img, top_left_x, top_left_y, PATCH_SIZE)
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


#preds = Trained_model.predict(test_gen)

true = np.stack([ test_gen.__getitem__(i)[1] for i in range(test_gen.__len__())], axis=0)
X = np.stack([ test_gen.__getitem__(i)[0][:,:,:, :2].squeeze() for i in range(test_gen.__len__())], axis=0)

true = list(map(lambda x: np.round(x,1), true))
X = list(map(lambda x: np.round(x,2), X))

mean_snap = list(map(lambda x: np.round(np.median(x)*255,2),X))

acc_array = np.array(list(map(lambda x, y: calculate_metrics(x, np.expand_dims(y,2)), true, avg_pred)))
acc_array = pd.DataFrame(acc_array, columns=('rmse', 'mae', 'pnsr', 'ssim', 'ms_ssim'))
#acc_array['std'] = list(map(np.mean, preds_std))
#acc_array['std'] = std_pred
#acc_array['true'] = true
#acc_array['input'] = X
#acc_array['pred'] = list(map(lambda x: np.round(x,2), avg_pred))
acc_array['m_snap'] = mean_snap

acc_array['Date'] = list(map(lambda x: os.path.basename(x)[:-4], test_input))
#acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H_%M_%S")
acc_array['Date'] = pd.to_datetime(acc_array['Date'], format="%Y-%m-%d %H:%M:%S")

tide_wave_cond = pd.read_csv('data_CNN/Data_processed/Meta_df.csv')[['Date', 'bathy', 'Tide', 'Hs_m', 'Tp_m', 'Dir_m']]
tide_wave_cond['Date']= pd.to_datetime(tide_wave_cond['Date'], format="%Y-%m-%d %H:%M:%S")

acc_array = pd.merge(acc_array, tide_wave_cond, on='Date', how='inner').drop_duplicates('rmse', ignore_index=True)


plt.scatter(acc_array['rmse'], acc_array['m_snap'])
plt.xlabel('RMSE')
plt.ylabel('Pixel intensity')

acc_array.mean()

# Rmse depending on tide
acc_array['rip'] = 0
acc_array.loc[acc_array['bathy'].isin( ['2017-03-27', '2018-01-31']), 'rip'] = 1


acc_array.to_csv('data_CNN/Results/Accuracy_data_ext_Unet.csv')

########################################
# Error map vs Tide                    #
########################################

bathy = '2017-03-27'
sel_df = acc_array[acc_array['bathy'] == '2017-03-27']
sel_df['Err'] = np.abs(sel_df['true'] - sel_df['pred'])

i = 0

def color_vec(df, i):
    vec = np.repeat('blue', len(df))
    vec[i] = 'red'
    return(vec)

plt.imshow(sel_df['input'][i][:,:,0]*-1, cmap='Greys')
plt.imshow(sel_df['input'][i][:,:,1]*-1, cmap='Greys')
plt.scatter('Date', 'Tide', data = sel_df, c=color_vec(sel_df,i))

_vmin, _vmax = np.min(sel_df['true'][i])-1, np.max(sel_df['true'][i])+1
plt.imshow(sel_df['true'][i], cmap='jet', vmin=_vmin, vmax=_vmax)
plt.imshow(sel_df['pred'][i], cmap='jet', vmin=_vmin, vmax=_vmax)
plt.imshow(sel_df['Err'][i], cmap='inferno')


import pickle

with open('df.pickle', 'wb') as handle:
    pickle.dump(acc_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
