#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python script to prepare the data to be used by keras models. The repository
 structure is specifically created to work with keras data loading procedures.

Usage:
    python3 generate_dataset.py [metadata.csv] [Output_dir]

    where [metadata.csv] is the file with all the metadata about the images.
    If not present in data_CNN/Data_processed repository, it will be created
    and stored in this repository. The[Output_dir] is the name of the repository
    where you want all the prepared data for the CNN training.

Author:
    Aurelien Callens - 27/06/2022
"""
import argparse
from os.path import exists

import configs.Settings_data as param_data
from src.data_prep.make_cnn_dataset import generate_data_folders_cnn
from src.data_prep.make_img_csv import make_img_csv


class Generate_dataset():
    """
    Custom class to generate dataset from metadata file

    ...

    Attributes
    ----------
    csv_path : str
        Path of the metadata csv. If the file does not exist, it creates the 
        metadata csv at this location
    fp_name : str
        Filepath of the output folder
    Parameters :
        All the others attributes are parameters imported from
        ./configs/Settings_data.py

    Methods
    -------
    generate_data_csv()
        Generate metadata csv from a parameter file: ./configs/Settings_data.py

    generate_cnn_dataset()
        Generate the folders with processed data. The structure of the folders
        is specifically created to work with keras data loading procedures.
    """
    def __init__(self, meta_csv, output_fp):

        self.csv_path = meta_csv
        self.fp_name = output_fp

        self.Img_folder_path = param_data.Img_folder_path
        self.wind_pos = param_data.wind_pos
        self.wind_pos_062021 = param_data.wind_pos_062021
        self.filt_rip = param_data.filt_rip
        self.df_fp_bat = param_data.df_fp_bat
        self.bathy_range = param_data.bathy_range
        self.output_size = param_data.output_size
        self.tide_min = param_data.tide_min
        self.tide_max = param_data.tide_max
        self.test_bathy = param_data.test_bathy

        if (not exists(meta_csv)):
            print('No existing meta csv. Creating one...')
            self.generate_data_csv()
        else:
            print('Reading existing meta csv. Creating dataset...')
        self.generate_cnn_dataset()

    def generate_data_csv(self):

        make_img_csv(self.csv_path, self.Img_folder_path,
                     self.wind_pos, self.wind_pos_062021,
                     self.bathy_range)

    def generate_cnn_dataset(self):

        generate_data_folders_cnn(self.fp_name, self.csv_path,
                                  self.df_fp_bat,
                                  self.output_size, self.tide_min,
                                  self.tide_max, self.test_bathy,
                                  self.filt_rip)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('meta_file', help="Meta_file")
    parser.add_argument('output_fp', help="Path for datafile")
    args = parser.parse_args()

    meta_file = 'data_CNN/Data_processed/' + args.meta_file
    output_dir = 'data_CNN/' + args.output_fp + '/'

    Generate_dataset(meta_csv = meta_file, output_fp = output_dir)

    print("Generation du dataset fini!")
