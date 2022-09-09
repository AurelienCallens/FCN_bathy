#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the data for training tensorflow models.

Usage:
    from src.data_prep.make_cnn_dataset import generate_data_folders_cnn

Author:
    Aurélien Callens - 05/05/2022
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import Point
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler

from src.data_prep.img_processing import img_rotation, proj_rot, crop_img, ffill


def generate_data_folders_cnn(fp_name, df_fp_img, df_fp_bat, output_size,
                              tide_min, tide_max, test_bathy, filt_rip):
    """Prepare the images and generate associated folders to train a tensorflow
    model

    This function prepares the images and the bathymetric data to be used by a
    tensorflow model. X's are arrays with 3 channels: mean RGB of snap, mean
    RGB of timex and matrix with normalized environmental conditions. Y's are
    array with 1 channel containing the bathymetric map to predict.
    The main steps of the function are:
        1- Read csv file with image characteristics and filepaths + csv file
        with bathymetric map
        2- Filter if necessary, split the data into train/val/test and
        normalize environmental conditions
        3- Create the directories to save all the X and Y. The structure of
        the directories is adapted for the use keras generator.
        4- Iterate through all the date to prepare the dataset
            a- Prepare X tensor
            b- Prepare Y tensor
            c- Save X and Y as numpy file in the appropriate directories

    Parameters
    ----------
    fp_name : str
        Name of the output directory
    df_fp_img : str
        Filepath of the repository containing the orthorectified images
    df_fp_bat : str
        Filepath of the csvfile with bathymetric data
    output_size : tuple
        Output size in pixels
    filt_rip : bool
        Filter images of the 2 bathy surveys with rip by tide. This removes 
        the images with no information i.e. where the tide is too high (<2.5m)
    tide_min : float
        Minimum tide level in meters to filter images by tide
    tide_max : float
        Maximum tide level in meters to filter images by tide
    test_bathy : str
        How to make the test set. If None: split the data 80% train/20% test
        for all the bathy surveys. If "2018-01-31": keeps "2018-01-31" survey
        as test data

    Output
    ------
    A single directory at the specified location. This directory is populated
    with subdirectories (train/val/test) containing X and Y data ready for the
    training of a tensorflow model.
    """
    # Import df
    # Img dataframe
    print('Importing meta csv')
    final_df = pd.read_csv(df_fp_img)
    if type(tide_min) == float:
        final_df = final_df[final_df['Z_m'] > tide_min].reset_index(drop=True)
    if type(tide_max) == float:
        final_df = final_df[final_df['Z_m'] < tide_max].reset_index(drop=True)

    if filt_rip:
        bathy_rip = ((final_df['bathy'] == '2017-03-27') | (final_df['bathy'] == '2018-01-31')) & (final_df['Tide'] > 2.5)
        final_df = final_df[~bathy_rip]

    final_df.sort_values('Date', ignore_index=True, inplace=True)

    final_df['Date'] = final_df['Date'].astype(str)

    if test_bathy is None:
        # Train test by day
        # Train
        final_df['Split'] = 'Train'

        # Test
        day_test = ['2017-03-28', '2018-01-31', '2021-03-03', '2021-06-20']
        final_df.loc[final_df['Date'].apply(lambda x: x[:10] in day_test), 'Split'] = 'Test'

        # Validation
        ind = final_df[final_df['Split'] == 'Train'].groupby('bathy', group_keys=False).apply(lambda x: x.sample(frac=0.2)).index
        final_df.loc[ind, 'Split'] = 'Validation'

    elif test_bathy == '2017-03-27':
        # Train test by bathy
        # Train
        final_df['Split'] = 'Train'

        # Test
        bathy_test = '2017-03-27'
        final_df.loc[final_df['bathy'].apply(lambda x: x == bathy_test), 'Split'] = 'Test'

        # Validation
        ind = final_df[final_df['Split'] == 'Train'].groupby('bathy', group_keys=False).apply(lambda x: x.sample(frac=0.2)).index
        final_df.loc[ind, 'Split'] = 'Validation'

    elif test_bathy == '2018-01-31':
        # Train test by bathy
        # Train
        final_df['Split'] = 'Train'

        # Test
        bathy_test = '2018-01-31'
        final_df.loc[final_df['bathy'].apply(lambda x: x == bathy_test), 'Split'] = 'Test'

        # Validation
        ind = final_df[final_df['Split'] == 'Train'].groupby('bathy', group_keys=False).apply(lambda x: x.sample(frac=0.2)).index
        final_df.loc[ind, 'Split'] = 'Validation'
    else:
        print("Wrong bathy for test")
        return

    # Scaling 0-1
    scaler = MinMaxScaler()
    final_df['Hs_c'] = scaler.fit_transform(final_df[['Hs_m']])
    final_df['Tp_c'] = scaler.fit_transform(final_df[['Tp_m']])
    final_df['Dir_c'] = scaler.fit_transform(final_df[['Dir_m']])
    final_df['Tide_c'] = scaler.fit_transform(final_df[['Tide']])

    # Extract unique date + hour
    date_unique = final_df['Date'].sort_values().unique()

    csv_name = 'data_CNN/Data_processed/Meta_df_' + os.path.basename(fp_name[:len(fp_name)-1]) + '.csv'
    final_df.to_csv(csv_name, index=False)

    # Bathy dataframe
    bat_df = pd.read_csv(df_fp_bat)
    print('Creating folders')
    # Create folders
    splits = ['Train', 'Validation', 'Test']
    target_paths = [fp_name + i + '/Target/' for i in splits]
    input_paths = [fp_name + i + '/Input/' for i in splits]
    newpaths = target_paths + input_paths
    list(map(lambda x: os.makedirs(x, exist_ok=True), newpaths))

    # Loop through every date + hour

    for date in date_unique:
        temp_df = final_df[final_df['Date'] == date].reset_index()

        # X tensor (3, img_size, img_size) with mean RGB snap, timex
        # and env. cond
        # Prepare snap and timex
        fp_snp = list(temp_df.loc[(temp_df['Type_img'] == 'snap'), 'Fp_img'])
        fp_tmx = list(temp_df.loc[(temp_df['Type_img'] == 'timex'), 'Fp_img'])

        data_snp, trans_snp = img_rotation(fp_snp[0])
        data_tmx, trans_tmx = img_rotation(fp_tmx[0])

        # show(np.mean(data_snp, axis=0), transform=transform_snp,
        # cmap='Greys_r')
        # show(np.mean(data_tmx, axis=0), transform=transform_tmx,
        # cmap='Greys_r')

        # Extract window
        win_size = output_size/2
        win_coord = eval(temp_df.loc[0, 'X_Y'])
        snp_mat = crop_img(data_snp, trans_snp, win_coord, win_size)
        tmx_mat = crop_img(data_tmx, trans_tmx, win_coord, win_size)

        # Prepare env cond matrix
        env_mat = np.zeros(snp_mat.shape)
        mid = int(snp_mat.shape[0]/2)
        env_mat[0:mid, 0:mid] = temp_df.loc[0, 'Hs_c']
        env_mat[0:mid, mid:] = temp_df.loc[0, 'Tp_c']
        env_mat[mid:, 0:mid] = temp_df.loc[0, 'Dir_c']
        env_mat[mid:, mid:] = temp_df.loc[0, 'Tide_c']

        # Prepare bathymetric data on the same window as X tensor
        bathy_survey = temp_df.loc[0, 'bathy']
        bat = bat_df[['x', 'y', bathy_survey]]
        dst_crs = CRS.from_proj4(proj_rot)
        pts = list(map(lambda x: Point(x), np.array(bat[['x', 'y']])))
        gdf_g = gpd.GeoDataFrame(bat, geometry=pts, crs=dst_crs)

        gdf_g[['x_n', 'y_n']] = gdf_g[['x', 'y']]
        x_arr = np.arange(win_coord[0], win_coord[0] + win_size, 0.5)
        y_arr = np.arange(win_coord[1], win_coord[1] + win_size, 0.5)
        x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
        z_mesh = griddata((gdf_g['x_n'], gdf_g['y_n']), gdf_g[bathy_survey],
                          (x_mesh, y_mesh), method='linear')
        z_mesh = np.flipud(z_mesh)

        # Solving the problem of missing value for the bathymetric survey of
        # 06/2021:
        if(np.isnan(z_mesh).any()):
            z_mesh = ffill(z_mesh)
            print('Filling values ...')

        # Export matrix as img
        mat_3d = np.dstack((snp_mat, tmx_mat, env_mat))
        name_inp = fp_name + temp_df.loc[0, 'Split'] + '/Input/' + date + '.npy'
        np.save(name_inp, mat_3d.astype(np.float16))

        name_tar = fp_name + temp_df.loc[0, 'Split'] + '/Target/' + date + '.npy'
        np.save(name_tar, z_mesh.astype(np.float16))

