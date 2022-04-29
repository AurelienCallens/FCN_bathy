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
from dataloader.CustomGenerator import CustomGenerator
from dataloader.seq_iterator_gan import ParallelIterator

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
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

        self.test_gen = CustomGenerator(batch_size=self.batch_size,
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

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch, batchIndex, originalBatchIndex, xAndY in ParallelIterator(
                                       self.train_gen,
                                       epochs,
                                       shuffle=False,
                                       use_on_epoch_end=True,
                                       workers = 8,
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
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    batchIndex, self.train_gen.__len__(),
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))

            # If at save interval => save generated image samples
            #if batchIndex % sample_interval == 0:
            #    self.sample_images(epoch, batchIndex)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.test_gen.__getitem__(epoch)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

network = Pix2Pix()
network.discriminator.summary()
network.generator.summary()
network.batch_size

network.train(100, batch_size=1, sample_interval=50)
test_img = network.test_gen.__getitem__(2)
img = network.generator.predict(test_img[0])
plt.imshow(img[0].squeeze(), cmap = 'jet')
plt.imshow(test_img [1][0][:,:,0].astype(np.float32), cmap = 'jet')
