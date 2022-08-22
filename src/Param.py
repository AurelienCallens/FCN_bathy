#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for importing hyperparameters from json file to dictionnary

Usage:
    from src.models.Pix2Pix import Pix2Pix

Author:
    Aurelien Callens - 22/06/2022
    Greatly inspired from : https://zerowithdot.com/json-for-ml/
"""

import json


class Param():

    """
    Class for building and training Pix2pix network given certain parameters.


    Attributes
    ----------
    filepath : str
        Filepath of the json file containing the hyperparameters

    Methods
    -------
    load()
        Load the hyperparameters
    make_new_json_file()
        Make initial json file
    """

    def __init__(self, filepath):

        try:
            p = {}
            with open(filepath) as f:
                self.p = json.load(f)
        except IOError:
            print("File not found.")
        except ValueError:
            print("File could not be read.")
        else:
            for key in p:
                self.h = p[key]
            print("Hyperparameters loaded OK.")

    def load(self):
        return(self.p)

    def make_new_json_file(self, filepath):
        """
        Make a new json file with specific structure

        Parameters
        ----------
        filepath : str
            where to save the json file containing the hyperparameters
        """
        hyper = {}
        hyper['Input'] = {'DIR_NAME': 'Data_ext',
                          'IMG_SIZE': '(512, 512)',
                          'N_CHANNELS': 3}
        hyper['Data_aug'] = {'BRIGHT_R': '[0.9, 1.1]',
                             'SHIFT_R': 0.1,
                             'ROT_R': 0,
                             'V_FLIP':  True,
                             'H_FLIP': True}
        hyper['Net_str'] = {'ACTIV': 'relu',
                            'K_INIT': 'he_normal',
                            'PRETRAINED_W': False,
                            'FILTERS': 16,
                            'FILTERS_G': 16,
                            'FILTERS_D': 8,
                            'NOISE_STD': 0.05,
                            'DROP_RATE': 0.2}
        hyper['Train'] = {'BATCH_SIZE': 6,
                          'BATCH_SIZE_P': 1,
                          'EPOCHS': 100,
                          'LR': 0.002,
                          'LR_P2P': 0.0002,
                          'OPTI': 'Adam' #tf.keras.optimizers.Adam(0.0002, 0.5)
                          }
        hyper['Callbacks'] = {'DECAY_LR': True,
                              'FACTOR_DECAY': 0.8,
                              'N_EPOCHS_DECAY': 20,
                              'PATIENCE': 20,
                              'PATIENCE_P2P': 50}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(hyper, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    Param()