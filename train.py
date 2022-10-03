#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python script to train the networks (Unet, Pix2Pix) with a configuration file

Usage:
    python3 train.py [gpu] [Name of config file]

    where [gpu] is an option to perform the trainng with (1) or without (0)
    a gpu and [Name of config file] is the name of the config file to be used.
    The config file must be located in ./configs/

Author:
    Aurelien Callens - 20/08/2022
"""
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

from src.Param import Param
from src.models.UnetModel import UNet
from src.models.Pix2Pix import Pix2Pix
from src.evaluation.metric_functions import *
from src.executor.tf_init import start_tf_session

class Bathy_inv_network:
    """
    Custom class to train the network depending on a configuration file in 
    json format.

    Attributes
    ----------
    params_file : str
        Filepath of the configuration file
    gpu : 0 or 1
        If the gpu must be used or not

    Methods
    -------
    train_unet()
        Train a Unet network built with parameters indicated in the config file

    train_Pix2pix()
        Train Pix2Pix network built with parameters indicated in the config file

    save_model()
        Save the model and its performances on the test set in a csv file.

    load_model()
        Load a save network by indicating the filepath of the saved model
    """
    def __init__(self, params_file, gpu):
        self.params = Param(params_file).load()
        self.params_file = params_file
        if gpu:
            start_tf_session('gpu')
        else: 
            start_tf_session('cpu')

    def train_unet(self, check_gen=True):

        Unet_model = UNet(self.params)

        # Check results of train gen
        if(check_gen):
            Unet_model.verify_generators(n_img=5)
            #plot_output_generator(Unet_model.data_generator('Train'), n_img=1)

        # Train the model
        Trained_model = Unet_model.train()
        test_gen = Unet_model.data_generator('Test')
        self.save_model(Trained_model, test_gen, net="Unet")
        tf.keras.backend.clear_session()

    def train_Pix2pix(self):

        network = Pix2Pix(self.params)
        network.train(sample_interval=1, img_index=8)
        test_gen = network.test_gen
        self.save_model(network, test_gen, net="Pix2pix")
        tf.keras.backend.clear_session()

    def save_model(self, Trained_model, test_gen, net='Unet'):
        # Saving the model and its performance
        if (net == 'Unet'):
            epoch_tr = len(Trained_model[1].history['loss'])
            filters_g = self.params['Net_str']['FILTERS']
            filters_d = None
            opti = 'Adam'
            lr = self.params['Train']['LR']
            fac_dec = self.params['Callbacks']['FACTOR_DECAY']
            ep_dec = self.params['Callbacks']['N_EPOCHS_DECAY']
            early_s = self.params['Callbacks']['PATIENCE']
            batch_s = self.params['Train']['BATCH_SIZE']
            metrics = np.round(Trained_model[0].evaluate(test_gen), 4)
            model = Trained_model[0]
        else:
            epoch_tr = Trained_model.epoch_tr
            filters_g = self.params['Net_str']['FILTERS_G']
            filters_d = self.params['Net_str']['FILTERS_D']
            opti = 'Adam'
            lr = self.params['Train']['LR_P2P']
            fac_dec = None
            ep_dec = None
            early_s = self.params['Callbacks']['PATIENCE_P2P']
            batch_s = self.params['Train']['BATCH_SIZE_P']
            model = Trained_model.generator
            model.compile(optimizer=tf.keras.optimizers.Adam(lr, 0.5),
                                  loss='mse', metrics=[root_mean_squared_error, absolute_error, ssim, ms_ssim, pred_min, pred_max])
            metrics = np.round(model.evaluate(test_gen), 4)


        name = 'trained_models/{model}_{data_fp}_f{filters}_b{batch_size}_ep{epochs}_{date}'.format(model=net,
                                                                                                    data_fp=self.params['Input']['DIR_NAME'],
                                                                                                    filters=filters_g,
                                                                                                    batch_size=batch_s,
                                                                                                    epochs=epoch_tr,
                                                                                                    date=datetime.now().strftime("%d-%m-%Y_%H_%M"))


        res_dict = {'Name': name,
                    'Data': self.params['Input']['DIR_NAME'],
                    'Param_file': os.path.basename(self.params_file),
                    'Input_size': str(self.params['Input']['IMG_SIZE']),
                    'Brightness_r': str(self.params['Data_aug']['BRIGHT_R']),
                    'Shift_r': self.params['Data_aug']['SHIFT_R'],
                    'Rotation_r': self.params['Data_aug']['ROT_R'],
                    'V_flip': self.params['Data_aug']['V_FLIP'],
                    'H_flip': self.params['Data_aug']['H_FLIP'],
                    'Filters_G': filters_g,
                    'Filters_D': filters_d,
                    'Acti': self.params['Net_str']['ACTIV'],
                    'Dropout': self.params['Net_str']['DROP_RATE'],
                    'Opti': opti,
                    'Lr': lr,
                    'Factor_decay': fac_dec,
                    'Epoch decay': ep_dec,
                    'Early stop': early_s,
                    'Epoch_tr': epoch_tr,
                    'Rmse': metrics[1],
                    'Mae': metrics[2],
                    'Ssim': metrics[3],
                    'Ms_ssim': metrics[4],
                    'Pred_min': metrics[5],
                    'Pred_max': metrics[6]}

        tf.keras.models.save_model(model, name)

        df = pd.DataFrame(res_dict, index=[0])
        output_path = 'trained_models/Results_test.csv'
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

    def load_model(self, path_model):
        Trained_model = tf.keras.models.load_model(path_model,
                                                   custom_objects={'absolute_error':absolute_error,
                                                                   'rmse': root_mean_squared_error,
                                                                   'ssim': ssim,
                                                                   'ms-ssim': ms_ssim,
                                                                   'pred_min':pred_min,
                                                                   'pred_max':pred_max},
                                                   compile=False)

        Trained_model.compile(optimizer=OPTIMIZER, loss='mse', metrics=[root_mean_squared_error, absolute_error, ssim, ms_ssim, pred_min, pred_max])
        return(Trained_model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', help="Use GPU?", type=bool)
    parser.add_argument('config', help="Name of config file (must be placed in ./configs/")
    args = parser.parse_args()

    params_file = 'configs/' + args.config

    Bathy_inv = Bathy_inv_network(params_file, args.gpu)
    #Bathy_inv.train_unet(check_gen=False)
    Bathy_inv.train_Pix2pix()
    print("Entrainement fini!")
