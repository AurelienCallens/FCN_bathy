#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unet Model"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose, GaussianNoise
from tensorflow.keras.optimizers import Adam

from src.utils import initialize_file_path
from src.evaluation.metric_functions import *
from src.dataloader.CustomGenerator import CustomGenerator
from src.evaluation.CallbackClasses import TimingCallback, StepDecay
from src.verification.verif_functions import plot_output_generator, check_nan_generator

class UNet():
    """Unet Model Class"""

    def __init__(self, params):

        self.train_input, self.train_target = initialize_file_path(params['Input']['DIR_NAME'], 'Train')
        self.val_input, self.val_target = initialize_file_path(params['Input']['DIR_NAME'], 'Validation')
        self.test_input, self.test_target = initialize_file_path(params['Input']['DIR_NAME'], 'Test')
        self.IMG_SIZE = eval(params['Input']['IMG_SIZE'])
        self.BANDS = params['Input']['N_CHANNELS']

        self.ACTIV = params['Net_str']['ACTIV']
        self.K_INIT = params['Net_str']['K_INIT']
        self.FILTERS = params['Net_str']['FILTERS']
        self.NOISE_STD = params['Net_str']['NOISE_STD']
        self.DROP_RATE = params['Net_str']['DROP_RATE']

        self.BATCH_SIZE = params['Train']['BATCH_SIZE']
        self.EPOCHS = params['Train']['EPOCHS']
        self.LR = params['Train']['LR']

        self.DECAY_LR = params['Callbacks']['DECAY_LR']
        self.FACTOR_DECAY = params['Callbacks']['FACTOR_DECAY']
        self.N_EPOCHS_DECAY = params['Callbacks']['N_EPOCHS_DECAY']
        self.PATIENCE = params['Callbacks']['PATIENCE']
        self.params = params

    def data_generator(self, split):
        if split == 'Train':
            train_gen = CustomGenerator(batch_size=self.BATCH_SIZE,
                                        params=self.params, 
                                        input_img_paths=self.train_input,
                                        target_img_paths=self.train_target,
                                        split='Train')
            return(train_gen)
        if split == 'Train_no_aug':
            train_gen = CustomGenerator(batch_size=1,
                                        params=self.params,
                                        input_img_paths=self.train_input,
                                        target_img_paths=self.train_target,
                                        split='Train_no_aug')
            return(train_gen)
        if split == 'Validation':
            val_gen = CustomGenerator(batch_size=self.BATCH_SIZE,
                                      params=self.params,
                                      input_img_paths=self.val_input,
                                      target_img_paths=self.val_target,
                                      split='Validation')

            return(val_gen)
        else:
            test_gen = CustomGenerator(batch_size=1,
                                       params=self.params,
                                       input_img_paths=self.test_input,
                                       target_img_paths=self.test_target,
                                       split='Test')
            return(test_gen)

    def build(self):

        def conv_block(input_layer, filters, drop_out=True):
            conv = Conv2D(filters=filters, **conv_args)(input_layer)
            conv = BatchNormalization(trainable=True)(conv)
            conv = Conv2D(filters=filters, **conv_args)(conv)
            conv = BatchNormalization(trainable=True)(conv)
            conv = GaussianNoise(self.NOISE_STD)(conv)
            if drop_out:
                conv = Dropout(self.DROP_RATE)(conv, training=True)
            return conv

        def encoder_block(input_layer, num_filters, drop_out=True):
            conv_b = conv_block(input_layer, num_filters, drop_out)
            pool = MaxPooling2D((2, 2))(conv_b)
            return conv_b, pool

        def decoder_block(input_layer, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2,
                                padding="same")(input_layer)
            x = BatchNormalization(trainable=True)(x)
            x = GaussianNoise(self.NOISE_STD)(x)
            x = Dropout(self.DROP_RATE)(x, training=True)
            x = concatenate([x, skip_features], axis=3)
            x = conv_block(x, num_filters, drop_out=False)
            return x

        conv_args = {"kernel_size": 3,
                     "activation": self.ACTIV,
                     "padding": 'same',
                     "kernel_initializer": self.K_INIT
                     }

        inputs = Input((self.IMG_SIZE[0], self.IMG_SIZE[1], self.BANDS))

        s1, p1 = encoder_block(inputs, self.FILTERS, drop_out=False)
        s2, p2 = encoder_block(p1, self.FILTERS*2, drop_out=True)
        s3, p3 = encoder_block(p2, self.FILTERS*4, drop_out=True)
        s4, p4 = encoder_block(p3, self.FILTERS*8, drop_out=True)

        b1 = conv_block(p4, self.FILTERS*16, drop_out=True)

        d1 = decoder_block(b1, s4, self.FILTERS*8)
        d2 = decoder_block(d1, s3, self.FILTERS*4)
        d3 = decoder_block(d2, s2, self.FILTERS*2)
        d4 = decoder_block(d3, s1, self.FILTERS)

        outputs = Conv2D(1, 1, padding="same", activation=None)(d4)

        model = Model(inputs, outputs, name="U-Net")
        model.compile(optimizer=Adam(self.LR), loss='mse',
                      metrics=[root_mean_squared_error, absolute_error, ssim, ms_ssim, pred_min, pred_max])

        return(model)

    def callback(self):
        earlystop = EarlyStopping(patience=self.PATIENCE)
        cb = TimingCallback()
        Checkpoint = ModelCheckpoint("trained_models/U_net.h5", save_best_only=True)
        if self.DECAY_LR:
            schedule = StepDecay(initAlpha=self.LR, factor=self.FACTOR_DECAY,
                                 dropEvery=self.N_EPOCHS_DECAY)
            return([cb, earlystop, Checkpoint, LearningRateScheduler(schedule)])
        else:
            return([cb, earlystop, Checkpoint])

    def train(self):
        # Generators
        train_generator = self.data_generator('Train')
        val_generator = self.data_generator('Validation')
        model = self.build()
        # Train the network
        history = model.fit(
            train_generator,
            epochs=self.EPOCHS,
            validation_data=val_generator,
            callbacks=self.callback())
        tf.keras.backend.clear_session()
        self.model = model
        return([model, history])

    def predict(self):
        test_generator = self.data_generator(split='Test')
        preds = self.model.predict(test_generator)
        return(preds)

    def verify_generators(self, n_img):
        test_gen = self.data_generator('Test')
        val_gen = self.data_generator('Validation')
        train_gen = self.data_generator('Train')

        print("Images flowing from: " + dir_name + "\n")
        print("Length of train generator: %d / Batch size: %d / Total items: %d \n" %
              (train_gen.__len__(),  batch_size, train_gen.__len__() * batch_size))
        print("Length of validation generator: %d / Batch size: %d / Total items: %d \n" %
              (val_gen.__len__(),  batch_size, val_gen.__len__() * batch_size))
        print("Length of test generator: %d / Batch size: %d / Total items: %d \n" %
              (test_gen.__len__(),  1, test_gen.__len__()))
        print("Checking for Nan values:\n")
        print('Train generator:')
        check_nan_generator(train_gen)
        print('Val generator:')
        check_nan_generator(val_gen)
        print('Test generator:')
        check_nan_generator(test_gen)

        print("Plotting %d images from train generator ....\n"%(n_img))
        plot_output_generator(train_gen, n_img=n_img)
