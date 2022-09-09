#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:27:39 2022

@author: aurelien
"""
# Settings for making data csv

# Filepaths

# Filepath for the orthorectified image repository:
Img_folder_path = '/home/aurelien/Desktop/Orthorectification/Ortho_CNN/'


# Settings for window positioning
# Window position for all the bathymetry except 06/2021:
wind_pos = '(70, 70)'
# Window position for 06/2021 bathymetry (because truncated bathymetry near the shore):
wind_pos_062021 = '(80, 100)'


# Settings for making data folders

# Filepaths
# Filepath of the csvfile with bathymetric data:
df_fp_bat = "./data_CNN/Data_processed/Processed_bathy.csv"

#Output size in pixels
output_size = 512


# Filtering by tide:
tide_min = 0.9
tide_max = None

# Filter images with tide above 2.5meters for bathy survey with RIP
filt_rip = True

# Keeping only the days before and after the bathymetric surveys:

"""
# Extended range
bathy_range = [["2017/03/24", "2017/03/31"],
               ["2018/01/28", "2018/02/03"],
               ["2021/02/28", "2021/03/06"],
               ["2021/06/18", "2021/06/24"]]
"""
# Strict range depending on wave conditions:
bathy_range = [["2017/03/25", "2017/03/29"],
               ["2018/01/28", "2018/01/31"],
               ["2021/03/01", "2021/03/05"],
               ["2021/06/19", "2021/06/21"]]

# Spliting the data:
# None : split the data 80% train /20% test for all the bathy surveys
# "2018-01-31" : keep "2018-01-31" survey as test data
# test_bathy = None
test_bathy = "2018-01-31"
# test_bathy = "2017-03-27"