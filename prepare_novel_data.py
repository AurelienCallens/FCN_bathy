#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare novel data in the right format for the deep learning network. The
folder paths indicated in the  "if __name__ == '__main__'" section must be
 changed to correspond to the novel data.

Usage:
    python3 prepare_novel_data.py

Author:
    Aurélien Callens - 05/05/2022
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import Point
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler

import configs.Settings_data as param_data
from src.data_prep.img_processing import img_rotation, proj_rot, crop_img, ffill

# img_folder = 'New_images/Ortho_new_images/'
# list_dl_img = 'New_images/img_09_2022.csv'
# meta_csv_name = 'New_images/Meta_09_2022.csv'

def make_meta_csv(meta_csv_name, img_folder, list_dl_img):
    """Make a csv file containing all the metadata about the novel images

    Parameters
    ----------
    meta_csv_name : str
        Filepath/name of the metadata file that will be created

    img_folder : str
        Filepath of the folder containing all the rectified images selected

    list_dl_img: str
        Csv file containing the metadata of novel data downloaded by scripts in
        SIRENA_downloader folders

    Output
    ------
    Save created table in a csv file with indicated name

    """

    # List all the images
    list_img = glob.glob(img_folder + 'biarritz_3_2_1_*.png')
    res_df = pd.DataFrame({'Fp_img': list_img})
    res_df['Bn_img'] = res_df['Fp_img'].apply(os.path.basename)
    res_df['Date'] = res_df['Bn_img'].apply(lambda x: x[15:34])
    res_df['Date'] = pd.to_datetime(res_df['Date'], format="%Y-%m-%d-%H-%M-%S")
    res_df['Date'] = res_df['Date'].apply(lambda x: x.replace(second=00))
    res_df['Type_img'] = res_df['Bn_img'].apply(lambda x: re.search(r'.*?ed\_(.*)\..*', x).group(1))
    res_df['Z_m'] = res_df['Bn_img'].apply(lambda x: re.search(r'.*?zplane\_(.*)\_bl.*', x).group(1).replace('_R', ''))

    # Find Tide level (joining by date)
    env_cond = pd.read_csv(list_dl_img)
    env_cond =  env_cond.rename(columns={'Hs': 'Hs_m', 'Tp': 'Tp_m', 'Dir':'Dir_m'})
    env_cond['Date'] = pd.to_datetime(env_cond['Date'], format="%Y-%m-%d %H:%M:%S")

    full_res = pd.merge(res_df, env_cond.groupby('Date').mean(),  on='Date', how='inner')
    full_res['X_Y'] = param_data.wind_pos
    full_res.to_csv(meta_csv_name)

def transform_test_image(fp_novel, fp_meta_csv, output_size):
    """Prepare the novel images to be used by a tensorflow
    model

    This function prepares the images and the bathymetric data to be used by a
    tensorflow model. X's are arrays with 3 channels: mean RGB of snap, mean
    RGB of timex and matrix with normalized environmental conditions.

    Parameters
    ----------
    fp_novel : str
        filepath of the csv file containing the metadata about the novel data

    fp_meta_csv : str
        filepath of the csv file containing the metadata about the original
        training data. It is needed to normalize the novel data.

    output_size : tuple
        Output size in pixels

    Output
    ------
    Save the processed input into a folder that can be exploited by keras models

    """
    # Import df
    # Img dataframe
    print('Importing csv of novel data')
    novel_df = pd.read_csv(fp_novel)
    novel_df.sort_values('Date', ignore_index=True, inplace=True)
    novel_df['Date'] = pd.to_datetime(novel_df['Date'], format="%Y-%m-%d %H:%M:%S")
    novel_df['Date'] = novel_df['Date'].apply(lambda x: x.strftime("%Y-%m-%d_%H-%M-%S"))
    novel_df['Date'] = novel_df['Date'].astype(str)

    print('Importing meta csv')
    ref_df = pd.read_csv(fp_meta_csv)

    # Scaling 0-1
    print('Normalizing env. cond. ...')

    scaler = MinMaxScaler()
    var_m = ['Hs_m', 'Tp_m', 'Dir_m', 'Tide']
    var_c = ['Hs_c', 'Tp_c', 'Dir_c', 'Tide_c']

    for i in range(len(var_m)):
        scaler.fit(ref_df[[var_m[i]]])
        novel_df[var_c[i]] = scaler.transform(novel_df[[var_m[i]]])

        if any(novel_df[var_c[i]] < 0) | any(novel_df[var_c[i]] > 1):
            print('Normalized values < 0 or > 1 for %s which indicates unseen env. cond. by the network!' % (var_m[i]))
            print('Training range for %s: %.2f - %.2f' % (var_m[i], min(ref_df[var_m[i]]), max(ref_df[var_m[i]])))
            print('Range for novel data: %.2f - %.2f' % (min(novel_df[var_m[i]]), max(novel_df[var_m[i]])))
            print('Clipping the values to stay in 0-1 interval. This may have a negative impact on the prediction!')
            novel_df.loc[novel_df[var_c[i]] < 0, var_c[i]] = 0
            novel_df.loc[novel_df[var_c[i]] > 1, var_c[i]] = 1

    # Bathy dataframe
    print('Creating folders')
    # Create folders
    splits = ['Test']
    input_paths = ["data_CNN/Test_data/" + i + '/Input/' for i in splits]
    newpaths = input_paths
    list(map(lambda x: os.makedirs(x, exist_ok=True), newpaths))

    # Extract unique date + hour
    date_unique = novel_df['Date'].sort_values().unique()

    # Loop through every date + hour

    for date in date_unique:
        print(date)
        temp_df = novel_df[novel_df['Date'] == date].reset_index()

        # X tensor (3, img_size, img_size) with mean RGB snap, timex
        # and env. cond
        # Prepare snap and timex
        fp_snp = list(temp_df.loc[(temp_df['Type_img'] == 'snap'), 'Fp_img'])
        fp_tmx = list(temp_df.loc[(temp_df['Type_img'] == 'timex'), 'Fp_img'])

        data_snp, trans_snp = img_rotation(fp_snp[0])
        data_tmx, trans_tmx = img_rotation(fp_tmx[0])

        # show(np.mean(data_snp, axis=0), transform=transform_snp, cmap='Greys_r')
        # show(np.mean(data_tmx, axis=0), transform=transform_tmx,
        # cmap='Greys_r')

        # Extract window
        win_size = output_size/2
        win_coord = eval(temp_df.loc[0, 'X_Y'])
        snp_mat = crop_img(data_snp, trans_snp, win_coord, win_size)/255
        tmx_mat = crop_img(data_tmx, trans_tmx, win_coord, win_size)/255

        # Prepare env cond matrix
        env_mat = np.zeros(snp_mat.shape)
        mid = int(snp_mat.shape[0]/2)
        env_mat[0:mid, 0:mid] = temp_df.loc[0, 'Hs_c']
        env_mat[0:mid, mid:] = temp_df.loc[0, 'Tp_c']
        env_mat[mid:, 0:mid] = temp_df.loc[0, 'Dir_c']
        env_mat[mid:, mid:] = temp_df.loc[0, 'Tide_c']

        # Export matrix as img
        mat_3d = np.dstack((snp_mat, tmx_mat, env_mat))
        name_inp = 'data_CNN/Test_data/' + 'Test' + '/Input/' + date + '.npy'
        np.save(name_inp, mat_3d.astype(np.float16))


if __name__ == '__main__':

    transform_test_image(fp_novel="./New_images/Meta_09_2022.csv",
                     fp_meta_csv="./data_CNN/Data_processed/Meta_df_extended.csv",
                     output_size=512)
    print('Novel data processed!')
