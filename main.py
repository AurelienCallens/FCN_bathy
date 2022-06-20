#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:43:32 2022

@author: Aurelien
"""
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from configs.Settings import *
from src.models.UnetModel import UNet
from src.models.Pix2Pix import Pix2Pix
from src.verification.verif_functions import *
from src.evaluation.metric_functions import *
from src.executor.tf_init import start_tf_session

class Bathy_inv_network:

    def __init__(self):
        start_tf_session(MODE, CPU_CORES)

    def train_unet(self, check_gen=True):

        Unet_model = UNet(size=IMG_SIZE, bands=N_CHANNELS)

        # Check results of train gen
        if(check_gen):
            Unet_model.verify_generators(n_img=5)
            #plot_output_generator(Unet_model.data_generator('Train'), n_img=1)

        # Train the model
        Trained_model = Unet_model.train()
        test_gen = Unet_model.data_generator('Test')
        self.save_model(Trained_model, test_gen, net="Unet")

    def train_Pix2pix(self, batch_size=6):

        network = Pix2Pix(batch_size)

        network.train(epochs=EPOCHS, sample_interval=1, img_index=8)
        Trained_model = network.generator
        test_gen = network.test_gen
        self.save_model(Trained_model, test_gen, net="Pix2pix")

    def save_model(self, Trained_model, test_gen, net='Unet'):
        # Saving the model and its performance
        if (net == 'Unet'):
            network = net
            epoch_tr = len(Trained_model[1].history['loss'])
            filters_g= FILTERS
            filters_d = None
            opti = 'Nadam'
            lr = LR
            fac_dec = DECAY_LR
            ep_dec = N_EPOCHS_DECAY
            early_s = PATIENCE
            metrics = np.round(Trained_model[0].evaluate(test_gen), 4)
            model = Trained_model[0]
        else:
            network = net
            epoch_tr = EPOCHS
            filters_g = FILTERS_G
            filters_d = FILTERS_D
            opti = 'Adam'
            lr = 0.0002
            fac_dec = None
            ep_dec = None
            early_s = None
            Trained_model.compile(optimizer=OPTI_P, loss='mse', metrics=[root_mean_squared_error, absolute_error, ssim, ms_ssim, pred_min, pred_max])
            metrics = np.round(Trained_model.evaluate(test_gen), 4)
            model = Trained_model

        name = 'trained_models/{model}_{data_fp}_f{filters}_b{batch_size}_ep{epochs}_{date}'.format(model=network,
                                                                                                    data_fp=DIR_NAME,
                                                                                                    filters=FILTERS,
                                                                                                    batch_size=BATCH_SIZE,
                                                                                                    epochs=epoch_tr,
                                                                                                    date=datetime.now().strftime("%d-%m-%Y"))


        res_dict = {'Name': name,
                    'Data': DIR_NAME,
                    'Input_size': str(IMG_SIZE),
                    'Brightness_r': str(BRIGHT_R),
                    'Shift_r': SHIFT_R,
                    'Rotation_r': ROT_R,
                    'V_flip': V_FLIP,
                    'H_flip': H_FLIP,
                    'Filters_G': filters_g,
                    'Filters_D': filters_d,
                    'Acti': ACTIV,
                    'Dropout': DROP_RATE,
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

"""
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

"""

if __name__ == '__main__':
    Bathy_inv = Bathy_inv_network()
    Bathy_inv.train_unet(check_gen=False)
    Bathy_inv.train_Pix2pix()
    print("Tu es le boss")
