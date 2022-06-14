#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:43:32 2022

@author: Aurelien
"""

import tensorflow as tf
from numpy.random import seed
from configs.Settings import *
from model.UnetModel import UNet
from model.Pix2Pix import Pix2Pix
from evaluation.verif_functions import *
from evaluation.metric_functions import *
from executor.tf_init import start_tf_session
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# 0) Initialize session
mode = 'cpu'
#mode = 'gpu'
start_tf_session(mode)

# keras seed fixing
seed(42)
# tensorflow seed fixing
tf.random.set_seed(42)

# 1) Unet model
Unet_model = UNet(size=img_size, bands=n_channels)

# Check FCN structure
model = Unet_model.build()
model.summary()

#check_nan_generator_unique(Unet_model.data_generator('Train'))
# Check results of train gen
# Unet_model.verify_generators(n_img=2)
plot_output_generator(Unet_model.data_generator('Train'), n_img=5)

# Train the model
Trained_model = Unet_model.train()

# Verification of the training loss
name = 'trained_models/Model_4'
plot_history(Trained_model[1])

preds = Unet_model.predict()
test_gen = Unet_model.data_generator('Test')
Trained_model[0].evaluate(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)

tf.keras.models.save_model(Trained_model[0], name)


# 2) Pix2pix network

network = Pix2Pix(batch_size=6)

# Check Pix2pix structure
network.discriminator.summary()
network.generator.summary()

# Check generated image
network.sample_images(10)

# Train the model
discri = network.discriminator.predict([network.test_gen.__getitem__(0)[1],
                               network.test_gen.__getitem__(0)[0]])

network.train(epochs=100, sample_interval=1, img_index=10)

# Save the model

tf.keras.models.save_model(network.generator, 'trained_models/cGAN_low_window_s0.9_noflip_2018')
Trained_model = tf.keras.models.load_model('trained_models/cGAN_low_window_s0.9_noflip_2018',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'rmse': root_mean_squared_error,
                                                           'ssim': ssim,
                                                           'ms-ssim': ms_ssim,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[root_mean_squared_error, absolute_error, ssim, ms_ssim, pred_min, pred_max])

Trained_model.evaluate(network.test_gen)

import os

for i in range(network.test_gen.__len__()):

    basename_file = os.path.basename(test_input_img_paths[i])
    imgs_B, imgs_A = network.test_gen.__getitem__(i)
    fake_A = Trained_model.predict(imgs_B)
    imgs_A = imgs_A.squeeze()
    imgs_B = imgs_B.squeeze()
    fake_A = fake_A.squeeze()

    _vmin, _vmax = np.min(imgs_A)-1, np.max(imgs_A) + 1

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=0.8, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, :2], )
    ax1.imshow(np.uint8(imgs_B[:, :, 0]*255), cmap='gray')
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(np.uint8(imgs_B[:, :, 1]*255), cmap='gray')
    ax3 = fig.add_subplot(gs[0, 4:])
    ax3.imshow(np.uint8(imgs_B[:, :, 2]*255), cmap='gray')
    ax4 = fig.add_subplot(gs[1, 0:3])
    im = ax4.imshow(imgs_A.astype('float32'), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax4)
    ax5 = fig.add_subplot(gs[1, 3:])
    im = ax5.imshow(gaussian_filter(fake_A.astype('float32'), sigma=6), cmap='jet', vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax5)
    ax1.title.set_text('Mean RGB Snap')
    ax2.title.set_text('Mean RGB Timex')
    ax3.title.set_text('Env. Cond.')
    ax4.title.set_text('True bathy')
    ax5.title.set_text('Pred. bathy')
    #plt.show()
    plt.savefig('Predictions_data_low_2018/' + basename_file + '.png')
    plt.close()

