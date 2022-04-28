#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:43:32 2022

@author: aurelien
"""
import tensorflow as tf
from numpy.random import seed
from configs.Settings import *
from model.UnetModel import UNet
from evaluation.verif_functions import *
from evaluation.metric_functions import *
from executor.tf_init import start_tf_session

# 0) Initialize session
mode = 'cpu'
#mode = 'gpu'
start_tf_session(mode)

# keras seed fixing 
seed(42)
# tensorflow seed fixing
tf.random.set_seed(42) 

# 1) Test one model
Unet_model = UNet(size=img_size, bands=n_channels)

# Check FCN structure
model = Unet_model.build()
model.summary()

# Check results of train gen
# check_nan_generator(Unet_model, split='Validation')
test_gen = Unet_model.data_generator('Test')
val_gen = Unet_model.data_generator('Validation')
train_gen = Unet_model.data_generator('Train')
test_gen.__len__()
train_gen.__len__() * batch_size
val_gen.__len__() * batch_size

plot_output_generator(Unet_model, n_img=2)


# Train the model
Trained_model = Unet_model.train()

# Verification of the training loss
name = 'trained_models/Model_1'
plot_history(Trained_model[1])

preds = Unet_model.predict()
test_gen = Unet_model.data_generator('Test')
Trained_model[0].evaluate(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)

tf.keras.models.save_model(Trained_model[0], name)


# 2) Load a model
Trained_model = tf.keras.models.load_model('trained_models/CNN_allbathy_newgen_batch6_filter16_lr0.001_decay_0.8_every40_activ_relu_bathy',
                                           custom_objects={'absolute_error':absolute_error,
                                                           'pred_min':pred_min,
                                                           'pred_max':pred_max},
                                           compile=False)

Trained_model.compile(optimizer=optimizer, loss='mse', metrics=[absolute_error, pred_min, pred_max])

test_gen = Unet_model.data_generator('Test')
Trained_model.evaluate(test_gen)
preds = Trained_model.predict(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)


