#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:27:39 2022

@author: aurelien
"""
# Settings for making data csv

# Filepaths
Img_folder_path = '/home/aurelien/Desktop/Orthorectification/Ortho_CNN_1/'
csv_path = './data_CNN/Data_processed/meta_df_ext.csv'

# Settings for window positioning
wind_pos = '(60, 60)'
wind_pos_062021 = '(80, 100)'


# Setting for making data folders

# Filepaths
fp_name = './data_CNN/Data_ext/'
df_fp_img = './data_CNN/Data_processed/meta_df_ext.csv'
df_fp_bat = "./data_CNN/Data_processed/Processed_bathy.csv"
output_size = 512


# Filtering
tide_min = None
tide_max = None
bathy_range = [["2017/03/24", "2017/03/31"],
               ["2018/01/28", "2018/02/03"],
               ["2021/02/28", "2021/03/06"],
               ["2021/06/18", "2021/06/24"]]

# Spliting
test_bathy = None  # "2018-01-31"
