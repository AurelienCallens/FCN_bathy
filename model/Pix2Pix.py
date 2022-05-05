#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pix2pix model mostly taken from : https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
"""

from __future__ import print_function, division


import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from configs.Settings import *
from evaluation.metric_functions import *
from dataloader.CustomGenerator import CustomGenerator
from dataloader.seq_iterator_gan import ParallelIterator

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, GaussianNoise
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class Pix2Pix():
    def __init__(self):

        # Input shape
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]
        self.channels = n_channels
        self.batch_size = 6
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data sequence generator
        self.dataset_name = 'test_res'
        self.train_gen = CustomGenerator(batch_size=self.batch_size,
                               img_size=img_size,
                               bands=self.channels,
                               input_img_paths=train_input_img_paths+val_input_img_paths,
                               target_img_paths=train_target_img_paths+val_target_img_paths,
                               split='Train')

        self.test_gen = CustomGenerator(batch_size=1,
                               img_size=img_size,
                               bands=self.channels,
                               input_img_paths=test_input_img_paths,
                               target_img_paths=test_target_img_paths,
                               split='Test')

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 8
        self.noise_std = 0.05
        self.drop_rate = 0.2
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=(512, 512, 1), name='target_image')
        img_B = Input(shape=(512, 512, 3), name='input_image')

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = concatenate([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(512, 512, 3))

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='linear')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = GaussianNoise(self.noise_std)(d)
            d = Dropout(self.drop_rate)(d)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=[512, 512, 1], name='target_image')
        img_B = Input(shape=[512, 512, 3], name='input_image')

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = concatenate([img_A, img_B])
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=5, img_index=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch, batchIndex, originalBatchIndex, xAndY in ParallelIterator(
                                       self.train_gen,
                                       epochs,
                                       shuffle=False,
                                       use_on_epoch_end=True,
                                       workers=8,
                                       queue_size=10):
            imgs_B, imgs_A = xAndY

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            if ((batchIndex + 1) == self.train_gen.__len__()) and (epoch % sample_interval == 0):
            #if ((batchIndex + 1) % 5 ==0):
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    batchIndex+1, self.train_gen.__len__(),
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))
                self.sample_images(img_ind=img_index)

    def sample_images(self, img_ind):

        imgs_B, imgs_A = self.test_gen.__getitem__(img_ind)
        fake_A = self.generator.predict(imgs_B)
        imgs_A = imgs_A.squeeze()
        imgs_B = imgs_B.squeeze()
        fake_A = fake_A.squeeze()

        _vmin, _vmax = np.min(imgs_A)-1, np.max(imgs_A)+1

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
        plt.show()
        

network = Pix2Pix()
network.discriminator.summary()
network.generator.summary()
network.batch_size
network.disc_patch
network.sample_images(11)



network.train(epochs=100, batch_size=6, sample_interval=1, img_index=45)
network.sample_images(6)
tf.keras.models.save_model(network.generator, 'cGAN_1_data_sup_1.1_bathy_01_31')

Trained_model = tf.keras.models.load_model('trained_models/cGAN_1_data_sup_1.1_bathy_01_31',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=Adam(0.0002, 0.5), loss='mse', metrics=[absolute_error, pred_min, pred_max])

Trained_model.evaluate(network.test_gen)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter


for i in range(network.test_gen.__len__()):

    basename_file = os.path.basename(test_input_img_paths[i])
    imgs_B, imgs_A = network.test_gen.__getitem__(i)
    fake_A = Trained_model.predict(imgs_B)
    imgs_A = imgs_A.squeeze()
    imgs_B = imgs_B.squeeze()
    fake_A = fake_A.squeeze()

    _vmin, _vmax = np.min(imgs_A)-1, np.max(imgs_A)+1

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
    plt.savefig('Predictions/' + basename_file + '.png')
    plt.close()



