#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to make a csv file from a reportory of timex and snap images"""


import os
import re
import glob
import imageio
import numpy as np
import pandas as pd
from datetime import timedelta
from itertools import chain
import matplotlib.pyplot as plt

# Settings
from configs.Settings_data import *

def make_img_csv(csv_path, Img_folder_path, wind_pos, wind_pos_062021):
    # List all the images
    list_img = glob.glob(Img_folder_path + 'biarritz_3_2_1_*.png')
    res_df = pd.DataFrame({'Fp_img': list_img})

    # Extract information from filename
    res_df['Bn_img'] = res_df['Fp_img'].apply(os.path.basename)
    res_df['Date'] = res_df['Bn_img'].apply(lambda x: x[15:34])
    res_df['Date'] = pd.to_datetime(res_df['Date'], format="%Y-%m-%d-%H-%M-%S")
    res_df['Date'] = res_df['Date'].apply(lambda x: x.replace(second=00))
    res_df['Type_img'] = res_df['Bn_img'].apply(lambda x: re.search(r'.*?ed\_(.*)\..*', x).group(1))
    res_df['Z_m'] = res_df['Bn_img'].apply(lambda x: re.search(r'.*?zplane\_(.*)\_bl.*', x).group(1).replace('_R', ''))

    # Find Tide level (joining by date)
    tide_data = pd.read_csv('./data_CNN/Data_processed/obs_tide.csv')

    tide_data['Date'] = pd.to_datetime(tide_data['Date'],
                                       format="%Y-%m-%d %H:%M:%S")
    tide_data.drop(['Obs_Tide', 'Tide'], axis=1, inplace=True)
    tide_data.rename(columns={'True_tide': 'Tide'}, inplace=True)

    new_df = pd.merge(res_df, tide_data, on='Date', how='left')
    # new_df['diff_tide'] = np.abs(new_df['Z_m'].astype(float) - new_df['Tide'])
    # new_df['diff_tide'].mean()
    new_df['Date_h'] = new_df['Date'].apply(lambda x: (x + timedelta(hours=1)).strftime("%Y-%m-%d %H"))

    # Find wave conditions (joining by date)
    wave_data = pd.read_csv('./data_CNN/Data_processed/wave_data_bathy.csv')
    wave_data['Date'] = pd.to_datetime(wave_data['Date'], format="%Y-%m-%dT%H:%M:%SZ")
    wave_data['Date_h'] = wave_data['Date'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
    wave_data.drop('VMDR', axis=1, inplace=True)
    wave_data['Dir_m'] = np.round(wave_data['Dir_m'], 1)

    final_df = pd.merge(new_df, wave_data.drop('Date', axis=1),
                        on='Date_h', how='left')

    # Associate bathymetric surveys

    # create a list of conditions
    conditions = [
        (final_df['Date'] < '2018-01-01'),
        (final_df['Date'] > '2018-01-01') & (final_df['Date'] < '2019-01-01'),
        (final_df['Date'] > '2018-01-01') & (final_df['Date'] < '2021-05-01'),
        (final_df['Date'] > '2021-05-01')
        ]
    # create a list of the values we want to assign for each condition
    values = ['2017-03-27', '2018-01-31', '2021-03-03', '2021-06-21']

    # create a new column and use np.select to assign values to it using our lists as arguments
    final_df['bathy'] = np.select(conditions, values)

    # Associate selected area for each Z_m
    final_df['X_Y'] = wind_pos
    ind = (final_df['bathy'] == '2021-06-21')
    final_df.loc[ind, 'X_Y'] = wind_pos_062021

    final_df.drop_duplicates(subset=['Bn_img'], inplace=True, ignore_index=True)

    # Filtering by luminosity
    fp_imgs = list(final_df['Fp_img'])
    mean_pix = []
    for fp_img in fp_imgs:
        data = imageio.imread(fp_img)
        data = data[:1000, 400:, :]
        plt.imshow(data)
        mean_pix.append(int(np.mean(data)))

    final_df['Lum_pix'] = mean_pix

    final_df = final_df.iloc[np.where(final_df.Lum_pix > 100)[0], :]

    # Remove date where we don't have snap+ timex
    date_unique = final_df['Date'].unique()
    date_verif = []
    for date in date_unique:
        temp_df = final_df[final_df['Date'] == date].reset_index()
        if temp_df.shape[0] == 2:
            date_verif.append(True)
        else:
            date_verif.append(False)

    final_df = final_df[final_df.Date.apply(lambda x: x in date_unique[date_verif])]

    final_df.to_csv(csv_path, index=False)
    print('Done')

make_img_csv(csv_path, Img_folder_path, wind_pos, wind_pos_062021)