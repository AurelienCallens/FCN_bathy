#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unet Model"""

import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose, GaussianNoise


from configs.Settings import *
from dataloader.Data_sequence_class import Custom_gen
from evaluation.Callback_classes import TimingCallback, StepDecay
from evaluation.Metric_functions import pixel_error, absolute_error, pred_min, pred_max


class UNet():
    """Unet Model Class"""
    def __init__(self, size, bands):
        self.img_rows = size[0]
        self.img_cols = size[1]
        self.img_size = size
        self.bands = bands
        self.batch_size = batch_size

    def data_generator(self, split):
        if split == 'Train':
            train_gen = Custom_gen(batch_size=self.batch_size,
                                   img_size=self.img_size,
                                   bands=self.bands,
                                   input_img_paths=train_input_img_paths,
                                   target_img_paths=train_target_img_paths,
                                   split='Train')

            val_gen = Custom_gen(batch_size=self.batch_size,
                                 img_size=self.img_size,
                                 bands=self.bands,
                                 input_img_paths=val_input_img_paths,
                                 target_img_paths=val_target_img_paths,
                                 split='Validation')

            return([train_gen, val_gen])
        else:
            test_gen = Custom_gen(batch_size=1,
                                  img_size=self.img_size,
                                  bands=self.bands,
                                  input_img_paths=test_input_img_paths,
                                  target_img_paths=test_target_img_paths,
                                  split='Test')
            return(test_gen)


    def build(self):

        conv_args = {"kernel_size": 3,
                     "activation": activ,
                     "padding": 'same',
                     "kernel_initializer": k_init
                     }

        inputs = Input((self.img_rows, self.img_cols, self.bands))

        # Conv 1
        conv1 = Conv2D(filters=filters, **conv_args)(inputs)
        conv1 = BatchNormalization(trainable=True)(conv1)
        conv1 = Conv2D(filters=filters, **conv_args)(conv1)
        conv1 = BatchNormalization(trainable=True)(conv1)
        conv1 = GaussianNoise(noise_std)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Conv 2
        conv2 = Conv2D(filters=filters*2, **conv_args)(pool1)
        conv2 = BatchNormalization(trainable=True)(conv2)
        conv2 = Conv2D(filters=filters*2, **conv_args)(conv2)
        conv2 = BatchNormalization(trainable=True)(conv2)
        conv2 = GaussianNoise(noise_std)(conv2)
        drop2 = Dropout(drop_rate)(conv2, training=True)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

        # Conv 3
        conv3 = Conv2D(filters=filters*4, **conv_args)(pool2)
        conv3 = BatchNormalization(trainable=True)(conv3)
        conv3 = Conv2D(filters=filters*4, **conv_args)(conv3)
        conv3 = BatchNormalization(trainable=True)(conv3)
        conv3 = GaussianNoise(noise_std)(conv3)
        drop3 = Dropout(drop_rate)(conv3, training=True)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

        # Conv 4
        conv4 = Conv2D(filters=filters*8, **conv_args)(pool3)
        conv4 = BatchNormalization(trainable=True)(conv4)
        conv4 = Conv2D(filters=filters*8, **conv_args)(conv4)
        conv4 = BatchNormalization(trainable=True)(conv4)
        conv4 = GaussianNoise(noise_std)(conv4)
        drop4 = Dropout(drop_rate)(conv4, training=True)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Conv 5
        conv5 = Conv2D(filters=filters*16, **conv_args)(pool4)
        conv5 = BatchNormalization(trainable=True)(conv5)
        conv5 = Conv2D(filters=filters*16, **conv_args)(conv5)
        conv5 = BatchNormalization(trainable=True)(conv5)
        drop5 = Dropout(0.5)(conv5, training=True)

        # UpConv 1
        conv5 = Conv2DTranspose(filters=filters*8, strides=(2, 2), **conv_args)((conv5))
        conv5 = BatchNormalization(trainable=True)(conv5)
        drop5 = Dropout(0.5)(conv5, training=True)
        merge6 = concatenate([drop4, drop5], axis=3)

        # UpConv 2
        conv6 = Conv2D(filters=filters*8, **conv_args)(merge6)
        conv6 = BatchNormalization(trainable=True)(conv6)
        conv6 = Conv2D(filters=filters*8, **conv_args)(conv6)
        conv6 = BatchNormalization(trainable=True)(conv6)
        conv6 = Conv2DTranspose(filters=filters*4, strides=(2, 2), **conv_args)((conv6))
        conv6 = BatchNormalization(trainable=True)(conv6)
        conv6 = GaussianNoise(noise_std)(conv6)
        drop6 = Dropout(drop_rate)(conv6, training=True)

        # UpConv 3
        merge7 = concatenate([drop3, drop6], axis=3)
        conv7 = Conv2D(filters=filters*4, **conv_args)(merge7)
        conv7 = BatchNormalization(trainable=True)(conv7)
        conv7 = Conv2D(filters=filters*4, **conv_args)(conv7)
        conv7 = BatchNormalization(trainable=True)(conv7)
        conv7 = Conv2DTranspose(filters=filters*2, strides=(2, 2), **conv_args)((conv7))
        conv7 = BatchNormalization(trainable=True)(conv7)
        conv7 = GaussianNoise(noise_std)(conv7)
        drop7 = Dropout(drop_rate)(conv7, training=True)

        # UpConv 4
        merge8 = concatenate([drop2, drop7], axis=3)
        conv8 = Conv2D(filters=filters*2, **conv_args)(merge8)
        conv8 = BatchNormalization(trainable=True)(conv8)
        conv8 = Conv2D(filters=filters*2, **conv_args)(conv8)
        conv8 = BatchNormalization(trainable=True)(conv8)
        conv8 = Conv2DTranspose(filters=filters, strides=(2, 2), **conv_args)((conv8))
        conv8 = BatchNormalization(trainable=True)(conv8)
        conv8 = GaussianNoise(noise_std)(conv8)
        drop8 = Dropout(drop_rate)(conv8, training=True)

        # Conv 5 output
        merge9 = concatenate([conv1, drop8], axis=3)
        conv9 = Conv2D(filters=filters, **conv_args)(merge9)
        conv9 = Conv2D(filters=filters, **conv_args)(conv9)
        conv9 = Conv2D(1, **conv_args)(conv9)
        conv10 = Conv2D(1, 1, activation=None)(conv9)

        model = Model(inputs=inputs, outputs=[conv10])
        model.compile(optimizer=optimizer, loss='mse', metrics=[absolute_error, pred_min, pred_max])

        return(model)

    def callback(self):
        earlystop = EarlyStopping(patience=epoch_patience)
        cb = TimingCallback()
        Checkpoint = ModelCheckpoint("trained_models/U_net.h5", save_best_only=True)
        if decaying_lr:
            schedule = StepDecay(initAlpha=initial_lr, factor=factor_decay, dropEvery=nb_epoch_decay)
            return([cb, earlystop, Checkpoint, LearningRateScheduler(schedule)])
        else:
            return([cb, earlystop, Checkpoint])

    def train(self):
        # Make generators object
        train_generator, val_generator = self.data_generator('Train')

        model = self.build()
        # Train the network
        history = model.fit(
            train_generator,
            epochs=n_epochs,
            validation_data=val_generator,
            callbacks=self.callback())
        tf.keras.backend.clear_session()
        self.model = model
        return([model, history])

    def predict(self):
        test_generator = self.data_generator(split='Test')
        preds = self.model.predict(test_generator)
        return(preds)
