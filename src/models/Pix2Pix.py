#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for Pix2pix model

Usage:
    from src.models.Pix2Pix import Pix2Pix

Author:
    Aurelien Callens - 14/04/2022
    Pix2pix class greatly inspired from:
        https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
"""

import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.layers import Input, Dropout, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, GaussianNoise

# Local import
from src.utils import initialize_file_path
from src.evaluation.metric_functions import *
from src.dataloader.CustomGenerator import CustomGenerator
from src.dataloader.seq_iterator_gan import ParallelIterator


class Pix2Pix():
    """
    Class for building and training Pix2pix network given certain parameters.

    ...

    Attributes
    ----------
    params : dict
        Parameters dictionnary imported with the Param() class. It contains
        all the hyperparameters needed to build and train the network.

    Methods
    -------
    build_generator()
        Build the generator model
    build_discriminator()
        Build the discriminator model
    train()
        Train the combined model by iterating through generator and
        discriminator training
    sample_images()
        Plot a sample image from the generator every # epochs. It allows to
        judge visually the quality of the images predicted by the generator.
    """

    def __init__(self, params):

        # Parameters
        # Inputs
        self.train_input, self.train_target = initialize_file_path(params['Input']['DIR_NAME'], 'Train')
        self.val_input, self.val_target = initialize_file_path(params['Input']['DIR_NAME'], 'Validation')
        self.test_input, self.test_target = initialize_file_path(params['Input']['DIR_NAME'], 'Test')
        self.IMG_SIZE = eval(params['Input']['IMG_SIZE'])
        self.IMG_ROWS = self.IMG_SIZE[0]
        self.IMG_COLS = self.IMG_SIZE[1]
        self.BANDS = params['Input']['N_CHANNELS']
        self.IMG_SHAPE = (self.IMG_ROWS, self.IMG_COLS, self.BANDS)

        # Network architectures
        self.gf = params['Net_str']['FILTERS_G']
        self.df = params['Net_str']['FILTERS_D']
        self.ACTIV = params['Net_str']['ACTIV']
        self.K_INIT = params['Net_str']['K_INIT']
        self.FILTERS = params['Net_str']['FILTERS']
        self.NOISE_STD = params['Net_str']['NOISE_STD']
        self.DROP_RATE = params['Net_str']['DROP_RATE']

        # Training
        self.EPOCHS = params['Train']['EPOCHS']
        self.BATCH_SIZE = params['Train']['BATCH_SIZE']
        self.LR = params['Train']['LR_P2P']
        self.PATIENCE = params['Callbacks']['PATIENCE_P2P']
        optimizer = Adam(self.LR, 0.5)
        optimizer_disc = SGD(0.0002)


        # Configure data sequence generator
        self.dataset_name = 'test_res'
        self.train_gen = CustomGenerator(batch_size=self.BATCH_SIZE,
                                         params=params,
                                         input_img_paths=self.train_input,
                                         target_img_paths=self.train_target,
                                         split='Train')
        
        self.val_gen = CustomGenerator(batch_size=self.BATCH_SIZE,
                                         params=params,
                                         input_img_paths=self.val_input,
                                         target_img_paths=self.val_target,
                                         split='Validation')

        self.test_gen = CustomGenerator(batch_size=1,
                                        params=params,
                                        input_img_paths=self.test_input,
                                        target_img_paths=self.test_target,
                                        split='Test')

        # Calculate output shape of D (PatchGAN)
        patch = int(self.IMG_ROWS / 2**4)
        self.disc_patch = (patch, patch, 1)


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_disc,
                                   metrics=['accuracy'],
                                   loss_weights=[0.5])


        # Build the generator
        self.generator = self.build_generator()
        # Input images and their conditioning images
        img_A = Input(shape=(self.IMG_ROWS, self.IMG_ROWS, 1), name='target_image')
        img_B = Input(shape=(self.IMG_ROWS, self.IMG_ROWS, self.BANDS), name='input_image')

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['binary_crossentropy', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """Build U-Net Generator"""

        conv_args = {"kernel_size": 4,
                     "activation": self.ACTIV,
                     "padding": 'same',
                     "kernel_initializer": self.K_INIT
                     }

        def conv2d(layer_input, filters, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, strides=2, **conv_args)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
                d = GaussianNoise(self.NOISE_STD)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters,strides=1, **conv_args)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = GaussianNoise(self.NOISE_STD)(u)
            u = concatenate([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(self.IMG_ROWS, self.IMG_ROWS, self.BANDS))

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
        """Build PatchGAN discriminator """

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = GaussianNoise(self.NOISE_STD)(d)
            d = Dropout(self.DROP_RATE)(d)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=[self.IMG_ROWS, self.IMG_ROWS, 1], name='target_image')
        img_B = Input(shape=[self.IMG_ROWS, self.IMG_ROWS, self.BANDS], name='input_image')

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = concatenate([img_A, img_B])
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        validity = Activation('sigmoid')(validity)
        return Model([img_A, img_B], validity)

    def train(self, sample_interval=5, img_index=1):
        """Train the pix2pix model

        Parameters
        ----------
        sample_interval: int
            Epoch interval at which a sample image is plotted
        img_index: int
            Index of the sample to be plotted at each interval
        """
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.BATCH_SIZE,) + self.disc_patch)
        fake = np.zeros((self.BATCH_SIZE,) + self.disc_patch)
        rmse = []
        best_rmse = 999
        for epoch, batchIndex, originalBatchIndex, xAndY in ParallelIterator(
                                       self.train_gen,
                                       self.EPOCHS,
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
            # d_loss = np.add(d_loss_real, d_loss_fake)/2

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            
            # Calculate metrics for early stopping
            current_model = self.generator
            current_model.compile(optimizer=Adam(self.LR, 0.5),
                                  loss='mse', metrics=[root_mean_squared_error, psnr])
            metrics = np.round(current_model.evaluate(self.val_gen, verbose=0), 4)
            rmse.append(metrics[1])
            
            if ((batchIndex + 1) == self.train_gen.__len__()) and ((epoch + 1) % self.PATIENCE == 0):
                mean_rmse = np.mean(rmse[-self.PATIENCE:])
                if (mean_rmse < best_rmse):
                        best_rmse = mean_rmse
                        print("New best MA rmse!")
                else:
                    print("MA rmse increasing: early stopping! Value: " + str(mean_rmse))
                    break
                
                    
            # Display progress at the end of each epoch
            if ((batchIndex + 1) == self.train_gen.__len__()) and (epoch % sample_interval == 0):

                print("[Epoch %d/%d] [Batch %d/%d] [D loss real: %f] [D loss fake: %f] [G loss: %f] [RMSE: %f] [MA RMSE: %f] time: %s" % (epoch+1, self.EPOCHS,
                                                                                                                                         batchIndex+1, self.train_gen.__len__(),
                                                                                                                                         d_loss_real[
                                                                                                                                             0],
                                                                                                                                         d_loss_fake[
                                                                                                                                             0],
                                                                                                                                         g_loss[0],
                                                                                                                                         metrics[1],
                                                                                                                                         best_rmse,
                                                                                                                                         elapsed_time))
                # Plot the progress                                                                                                                         
                self.sample_images(epoch, img_ind=img_index)
            

    def sample_images(self, epoch, img_ind):
        """Function to plot the predicted image from the generator

        Parameters
        ----------
        epoch: int
            Epoch number of the prediction
        img_ind: int
            Index of the sample to be plotted
        """

        imgs_B, imgs_A = self.test_gen.__getitem__(img_ind)
        fake_A = self.generator.predict(imgs_B)
        imgs_A = imgs_A.squeeze()
        imgs_B = imgs_B.squeeze()
        fake_A = fake_A.squeeze()

        _vmin, _vmax = np.min(imgs_A)-0.5, np.max(imgs_A)+0.5

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
        im = ax4.imshow(imgs_A.astype('float32'), cmap='jet', vmin=_vmin,
                        vmax=_vmax)
        plt.colorbar(im, ax=ax4)
        ax5 = fig.add_subplot(gs[1, 3:])
        im = ax5.imshow(gaussian_filter(fake_A.astype('float32'), sigma=2),
                        cmap='jet', vmin=_vmin, vmax=_vmax)
        plt.colorbar(im, ax=ax5)
        ax1.title.set_text('Mean RGB Snap')
        ax2.title.set_text('Mean RGB Timex')
        ax3.title.set_text('Env. Cond.')
        ax4.title.set_text('True bathy')
        ax5.title.set_text('Pred. bathy')
        fig.suptitle('Results after epoch ' + str(epoch+1))
        # fig.savefig('trained_models/example_ouptut_epoch' + str(epoch) + '.png')
        fig.savefig('trained_models/example_ouptut_epoch.png')
        plt.close(fig)
