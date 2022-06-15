#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unet Model"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose, GaussianNoise

from configs.Settings import *
from evaluation.metric_functions import *
from dataloader.CustomGenerator import CustomGenerator
from evaluation.CallbackClasses import TimingCallback, StepDecay
from evaluation.verif_functions import plot_output_generator, check_nan_generator

class UNet():
    """Unet Model Class"""

    def __init__(self, size, bands):
        self.img_rows = size[0]
        self.img_cols = size[1]
        self.img_size = size
        self.bands = bands
        self.batch_size = BATCH_SIZE

    def data_generator(self, split):
        if split == 'Train':
            train_gen = CustomGenerator(batch_size=self.batch_size,
                                   img_size=self.img_size,
                                   bands=self.bands,
                                   input_img_paths=train_input_img_paths,
                                   target_img_paths=train_target_img_paths,
                                   split='Train')
            return(train_gen)
        if split == 'Train_no_aug':
            train_gen = CustomGenerator(batch_size=1,
                                   img_size=self.img_size,
                                   bands=self.bands,
                                   input_img_paths=train_input_img_paths,
                                   target_img_paths=train_target_img_paths,
                                   split='Train_no_aug')
            return(train_gen)
        if split == 'Validation':
            val_gen = CustomGenerator(batch_size=self.batch_size,
                                 img_size=self.img_size,
                                 bands=self.bands,
                                 input_img_paths=val_input_img_paths,
                                 target_img_paths=val_target_img_paths,
                                 split='Validation')

            return(val_gen)
        else:
            test_gen = CustomGenerator(batch_size=1,
                                  img_size=self.img_size,
                                  bands=self.bands,
                                  input_img_paths=test_input_img_paths,
                                  target_img_paths=test_target_img_paths,
                                  split='Test')
            return(test_gen)

    def build(self):

        def conv_block(input_layer, filters, drop_out=True):
            conv = Conv2D(filters=filters, **conv_args)(input_layer)
            conv = BatchNormalization(trainable=True)(conv)
            conv = Conv2D(filters=filters, **conv_args)(conv)
            conv = BatchNormalization(trainable=True)(conv)
            conv = GaussianNoise(NOISE_STD)(conv)
            if drop_out:
                conv = Dropout(DROP_RATE)(conv, training=True)
            return conv

        def encoder_block(input_layer, num_filters, drop_out=True):
            conv_b = conv_block(input_layer, num_filters, drop_out)
            pool = MaxPooling2D((2, 2))(conv_b)
            return conv_b, pool

        def decoder_block(input_layer, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
            x = BatchNormalization(trainable=True)(x)
            x = GaussianNoise(NOISE_STD)(x)
            x = Dropout(DROP_RATE)(x, training=True)
            x = concatenate([x, skip_features], axis=3)
            x = conv_block(x, num_filters, drop_out=False)
            return x

        conv_args = {"kernel_size": 3,
                     "activation": ACTIV,
                     "padding": 'same',
                     "kernel_initializer": K_INIT
                     }

        inputs = Input((self.img_rows, self.img_cols, self.bands))

        s1, p1 = encoder_block(inputs, FILTERS, drop_out=False)
        s2, p2 = encoder_block(p1, FILTERS*2, drop_out=True)
        s3, p3 = encoder_block(p2, FILTERS*4, drop_out=True)
        s4, p4 = encoder_block(p3, FILTERS*8, drop_out=True)

        b1 = conv_block(p4, FILTERS*16, drop_out=True)

        d1 = decoder_block(b1, s4, FILTERS*8)
        d2 = decoder_block(d1, s3, FILTERS*4)
        d3 = decoder_block(d2, s2, FILTERS*2)
        d4 = decoder_block(d3, s1, FILTERS)

        outputs = Conv2D(1, 1, padding="same", activation=None)(d4)

        model = Model(inputs, outputs, name="U-Net")
        model.compile(optimizer=OPTIMIZER, loss='mse',
                      metrics=[absolute_error, pred_min, pred_max])

        return(model)

    def callback(self):
        earlystop = EarlyStopping(patience=PATIENCE)
        cb = TimingCallback()
        Checkpoint = ModelCheckpoint("trained_models/U_net.h5", save_best_only=True)
        if DECAY_LR:
            schedule = StepDecay(initAlpha=INITIAL_LR, factor=FACTOR_DECAY, dropEvery=N_EPOCHS_DECAY)
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
            epochs=EPOCHS,
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
