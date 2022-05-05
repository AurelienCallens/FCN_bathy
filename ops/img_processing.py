#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:36:54 2022

@author: aurelien
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform


def img_rotation(data_fp):
    data = rasterio.open(data_fp)
    bn = os.path.basename(data_fp)
    # Save as raster to facilitate processing
    raster_name = "./data_CNN/temp.tif"
    with rasterio.open(
        raster_name,
        'w',
        driver='GTiff',
        height=data.height,
        width=data.width,
        count=3,
        crs="+init=EPSG:3943",
        transform=data.transform,
        dtype='uint8'
    ) as dst:
        for i in range(1, data.count+1):
            dst.write(data.read(i), i)
        dst.close()
    # Open raster file
    data = rasterio.open(raster_name)
    # Reproject on new CRS with rotation
    proj = "+proj=omerc +lat_0=43.483179 +lonc=-1.560958 +alpha=-40 +k=1 +x_0=0 +y_0=0 +gamma=0 +datum=WGS84 +towgs84=0,0,0,0,0,0,0  +units=m +no_defs"
    with rasterio.Env():
        # Source file
        rows, cols = data.shape
        src_transform = data.transform
        src_crs = CRS.from_proj4("+init=EPSG:3943")
        source = rasterio.band(data, [1, 2, 3])
        # Destination file with new CRS (Rotation)
        dst_shape = data.shape
        dst_crs = CRS.from_proj4(proj)
        transform, width, height = calculate_default_transform(
            data.crs, dst_crs, data.width, data.height, *data.bounds)
        destination = np.zeros((3, data.shape[0], 2000), np.uint8)
        test = reproject(
            source,
            destination,
            src_transform=src_transform,
            src_crs=data.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
    return([test[0], transform])

proj = "+proj=omerc +lat_0=43.483179 +lonc=-1.560958 +alpha=-40 +k=1 +x_0=0 +y_0=0 +gamma=0 +datum=WGS84 +towgs84=0,0,0,0,0,0,0  +units=m +no_defs"

def crop_img(img_mat, img_trans, win_corner, win_size):

    #Define parameters
    coord_w = win_corner
    mean_img = np.mean(img_mat, axis=0)
    shp_img = mean_img.shape

    #Find true coordinates depending on transform parameters
    x = np.arange(img_trans[2],  shp_img[1]*img_trans[0] - np.abs(img_trans[2]), img_trans[0])
    y = np.arange(img_trans[5], (shp_img[0]*img_trans[4] + img_trans[5]), -img_trans[0])

    #Extract window
    x_ind = np.where((x > coord_w[0]) & (x < (coord_w[0]+win_size)))[0]
    y_ind = np.where((y > coord_w[1]) & (y < (coord_w[1]+win_size)))[0]
    mat = mean_img[(y_ind.min()-1):y_ind.max(), (x_ind.min()-1):x_ind.max()]

    return(mat)

# ffill along axis 1, as provided in the answer by Divakar
def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out