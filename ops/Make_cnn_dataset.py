#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the dataset for the FCN (train/test)"""

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import Point
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from ops.utils import img_rotation, proj, crop_img, ffill

# Settings
output_size = 512
fp_name = './data_CNN/Data_bathy_sup_1.1/'
df_fp_img = './data_CNN/Data_processed/meta_df.csv'
df_fp_bat = "./data_CNN/Data_processed/Processed_bathy.csv"
tide_min = 1.1

# Import df
## Img dataframe
final_df = pd.read_csv(df_fp_img)
if type(tide_min) == float:
    final_df = final_df[final_df['Z_m'] > tide_min].reset_index(drop=True)

final_df.sort_values('Date', ignore_index=True, inplace=True)

#### Train test by day 
##### Train
final_df['Split'] = 'Train'

##### Test
day_test = ['2017-03-28', '2018-01-31', '2021-03-03', '2021-06-21']
final_df.loc[final_df['Date'].apply(lambda x: x[:10] in day_test), 'Split'] = 'Test'

##### Validation
ind = final_df[final_df['Split'] == 'Train'].groupby('bathy', group_keys=False).apply(lambda x: x.sample(frac=0.2)).index
final_df.loc[ind, 'Split'] = 'Validation'

### Train/Test
#### Train test by bathy

##### Train
final_df['Split'] = 'Train'

##### Test
bathy_test = '2021-06-21'
final_df.loc[final_df['bathy'].apply(lambda x: x == bathy_test), 'Split'] = 'Test'

##### Validation
ind = final_df[final_df['Split'] == 'Train'].groupby('bathy', group_keys=False).apply(lambda x: x.sample(frac=0.2)).index
final_df.loc[ind, 'Split'] = 'Validation'



### Scaling 0-1
scaler = MinMaxScaler()
final_df['Hs_c'] = scaler.fit_transform(final_df[['Hs_m']])
final_df['Tp_c'] = scaler.fit_transform(final_df[['Tp_m']])
final_df['Dir_c'] = scaler.fit_transform(final_df[['Dir_m']])
final_df['Tide_c'] = scaler.fit_transform(final_df[['Tide']])


### Extract unique date
date_unique = final_df['Date'].sort_values().unique()

## Bathy dataframe
bat_df = pd.read_csv(df_fp_bat)

# Loop through every date

for date in date_unique:
    temp_df = final_df[final_df['Date'] == date].reset_index()

    # X tensor composed of size (3, 400, 400) with mean RGB snap, tmx and env. cond
    ## Prepare snap and timex
    fp_snp = list(temp_df.loc[(temp_df['Type_img'] == 'snap'), 'Fp_img'])
    fp_tmx = list(temp_df.loc[(temp_df['Type_img'] == 'timex'), 'Fp_img'])

    data_snp, trans_snp = img_rotation(fp_snp[0])
    data_tmx, trans_tmx = img_rotation(fp_tmx[0])

    # show(np.mean(data_snp, axis=0), transform=transform_snp, cmap='Greys_r')
    # show(np.mean(data_tmx, axis=0), transform=transform_tmx, cmap='Greys_r')
    win_size = output_size/2
    win_coord = eval(temp_df.loc[0, 'X_Y'])
    snp_mat = crop_img(data_snp, trans_snp, win_coord, win_size)
    tmx_mat = crop_img(data_tmx, trans_tmx, win_coord, win_size)

    ## Prepare env cond matrix
    env_mat = np.zeros(snp_mat.shape)
    mid = int(snp_mat.shape[0]/2)
    env_mat[0:mid, 0:mid] = temp_df.loc[0, 'Hs_c']
    env_mat[0:mid, mid:] = temp_df.loc[0, 'Tp_c']
    env_mat[mid:, 0:mid] = temp_df.loc[0, 'Dir_c']
    env_mat[mid:, mid:] = temp_df.loc[0, 'Tide_c']

    ## Prepare bathymetric data
    bathy_survey = temp_df.loc[0, 'bathy']
    bat = bat_df[['x', 'y', bathy_survey]]
    dst_crs = CRS.from_proj4(proj)
    pts = list(map(lambda x: Point(x), np.array(bat[['x', 'y']])))
    gdf_g = gpd.GeoDataFrame(bat, geometry=pts, crs=dst_crs)

    gdf_g[['x_n', 'y_n']] = gdf_g[['x', 'y']]
    x_arr = np.arange(win_coord[0], win_coord[0] + win_size, 0.5)
    y_arr = np.arange(win_coord[1], win_coord[1] + win_size, 0.5)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    z_mesh = griddata((gdf_g['x_n'], gdf_g['y_n']), gdf_g[bathy_survey],
                      (x_mesh, y_mesh), method='linear')
    z_mesh = np.flipud(z_mesh)

    if(np.isnan(z_mesh).any() == True):
        z_mesh = ffill(z_mesh)
        print('Filling values ...')

    # Export matrix as img
    mat_3d = np.dstack((snp_mat, tmx_mat, env_mat))
    name_inp = fp_name + temp_df.loc[0, 'Split'] + '/Input/' + date + '.npy'
    np.save(name_inp, mat_3d.astype(np.float32))

    name_tar = fp_name + temp_df.loc[0, 'Split'] + '/Target/' + date + '.npy'
    np.save(name_tar, z_mesh.astype(np.float32))

    """
    np.save('out.npy', mat_3d.astype(np.float16))
    img_test = np.load('out.npy')
    plt.imshow(img_test)
    plt.subplot(2, 2, 1)
    plt.imshow(snp_mat, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(tmx_mat, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(env_mat)
    plt.subplot(2, 2, 4)
    plt.imshow(z_mesh)
    plt.show()
    """