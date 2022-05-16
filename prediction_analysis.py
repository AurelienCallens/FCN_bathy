#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:58:37 2022

@author: aurelien
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.random import seed
from configs.Settings import *
from model.UnetModel import UNet
from evaluation.verif_functions import *
from evaluation.metric_functions import *
from executor.tf_init import start_tf_session
import matplotlib.gridspec as gridspec
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

Trained_model = tf.keras.models.load_model('trained_models/cGAN_1_data_sup_1.1_noise_binary_loss',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[absolute_error, psnr, ssim, ms_ssim])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)

preds = Trained_model.predict(test_gen)


plot_predictions(test_generator=test_gen, predictions=preds, every_n=4)

pred_number = 10
item = 90
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
im = ax6.imshow(pred_0.std(axis=2)*2, cmap='inferno', vmin=0, vmax=(pred_0.std(axis=2)*2).max())
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
ax1.title.set_text('Input Snap')
ax2.title.set_text('Input Timex')
ax3.title.set_text('True bathy')
ax4.title.set_text('Pred. bathy')
ax5.title.set_text('Abs. Error')
ax6.title.set_text('Uncertainty')
plt.show()


## Grad Cam ? Grad Ram ?
#https://github.com/cauchyturing/kaggle_diabetic_RAM

Trained_model.summary()


from tensorflow.keras.models import Model
import tensorflow as tf, numpy as np, cv2

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError('Could not find 4D layer. Cannot apply GradCAM.')

    def compute_heatmap(self, image, eps=1e-08):
        gradModel = Model(inputs=[
         self.model.inputs],
          outputs=[
         self.model.get_layer(self.layerName).output,
         self.model.output])
        with tf.GradientTape() as (tape):
            inputs = tf.cast(image, tf.float32)
            convOutputs, predictions = gradModel(inputs)
            loss = predictions[:]
        grads = tape.gradient(loss, convOutputs)
        #castConvOutputs = tf.cast(convOutputs > 0, 'float32')
        castConvOutputs = tf.cast(convOutputs, 'float32')
        #castGrads = tf.cast(grads > 0, 'float32')
        castGrads = tf.cast(grads, 'float32')
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum((tf.multiply(weights, convOutputs)), axis=(-1))
        w, h = image.shape[2], image.shape[1]
        heatmap = cv2.resize(cam.numpy(), (w, h))
        #numer = heatmap - np.min(heatmap)
        #denom = heatmap.max() - heatmap.min() + eps
        #heatmap = numer / denom
        #heatmap = (heatmap * 255).astype('uint8')
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output= cv2.addWeighted(image.squeeze().astype('uint8'), alpha, heatmap, 1 - alpha, 0)
        return (output)

# initialize our gradient class activation map and build the heatmap
last_conv_layer_name = "conv2d_16"
cam = GradCAM(Trained_model, true_0, layerName=last_conv_layer_name)
heatmap = cam.compute_heatmap(input_0)



fig = plt.figure(figsize=(9, 8))
gs = gridspec.GridSpec(8, 8)
gs.update(wspace=1, hspace=1)
ax1 = fig.add_subplot(gs[:4, :4])
im = ax1.imshow(heatmap, cmap='jet')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax2 = fig.add_subplot(gs[:4, 4:])
ax2.imshow(np.uint8(input_0[0,:,:,0]*255), cmap='gray')
ax3 = fig.add_subplot(gs[4:, :4])
im = ax3.imshow(np.uint8(input_0[0,:,:,1]*255), cmap='gray')
ax4 = fig.add_subplot(gs[4:, 4:])
im = ax4.imshow(np.uint8(input_0[0,:,:,2]*255), cmap='gray')





heat = plt.imshow(heatmap, cmap='jet')
#plt.colorbar(heat, fraction=0.046, pad=0.04)
plt.imshow(input_0.squeeze().astype('float32'))


##### https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0


img = input_0[0]

def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:(top_left_y + patch_size), top_left_x:(top_left_x + patch_size), :] = 0

    return patched_image

plt.imshow(apply_grey_patch(img, 0, 0, 64).astype('float32'))

pred_0 = Trained_model.predict(input_0)
PATCH_SIZE =16
sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

# Iterate the patch over the image
for top_left_x in range(0, img.shape[0], PATCH_SIZE):
    for top_left_y in range(0, img.shape[1], PATCH_SIZE):
        patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)
        predicted_img = Trained_model.predict(np.array([patched_image]))[0]
        
        confidence = float(pixel_error(true_0, predicted_img)/pixel_error(true_0, pred_0))
        
        # Save confidence for this specific patched image in map
        sensitivity_map[
            top_left_y:top_left_y + PATCH_SIZE,
            top_left_x:top_left_x + PATCH_SIZE,
        ] = confidence




im=plt.imshow(gaussian_filter(sensitivity_map, 5) , cmap='jet')
plt.colorbar(im,  fraction=0.046, pad=0.04)

sensitivity_map.shape



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

acc_array.mean()
acc_array.hist()
acc_array[:, :2].argmax(axis=0)
acc_array[:, 2:].argmin(axis=0)

# Accuracy first bathy 

acc_array.loc[:34, :].mean()

acc_array.loc[35:56, :].mean()

acc_array.loc[57:90, :].mean()

acc_array.loc[91:, :].mean()
