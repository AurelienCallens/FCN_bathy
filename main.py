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

# 1) Unet model
Unet_model = UNet(size=img_size, bands=n_channels)

# Check FCN structure
model = Unet_model.build()
model.summary()

# Check results of train gen
#Unet_model.verify_generators(n_img=2)

# Train the model
Trained_model = Unet_model.train()

# Verification of the training loss
name = 'trained_models/Model_3'
plot_history(Trained_model[1])

preds = Unet_model.predict()
test_gen = Unet_model.data_generator('Test')
Trained_model[0].evaluate(test_gen)

plot_predictions(test_generator=test_gen, predictions=preds, every_n=2)

tf.keras.models.save_model(Trained_model[0], name)



