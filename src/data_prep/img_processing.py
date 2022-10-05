#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains functions to rotate and crop the orthorectified snap and timex.

Usage:
    from src.data_prep.img_processing import img_rotation, proj_rot, crop_img, ffill

Author:
    Aurélien Callens - 05/05/2022
"""

import os
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform

# Rotated projection
proj_rot = "+proj=omerc +lat_0=43.483179 +lonc=-1.560958 +alpha=-40 +k=1 +x_0=0 +y_0=0 +gamma=0 +datum=WGS84 +towgs84=0,0,0,0,0,0,0  +units=m +no_defs"


def img_rotation(data_fp):
    """Rotate the orthorectified images by -40 degrees in order to have nice
    image (shore at the bottom)

    First the function open the orthorectified image at the indicated path and
    transform it into a raster by saving it to a temporary file. Then it loads
    the raster from the temporary file and performs the rotation by reprojecting
    the raster. This function returns a list with the rotated image as array
    and the associated transform.

    Parameters
    ----------
    data_fp : str
        The filepath of the snap or timex

    Output
    ------
    List
        The first element of the list is the array of the rotated image
        and the second element is the associated transform.
    """
    # Read the image
    data = rasterio.open(data_fp)

    # Save as raster to facilitate processing (doesn't work otherwise)
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
    with rasterio.open(raster_name) as data:
        # Reproject on new CRS with rotation
        proj = proj_rot
        with rasterio.Env():
            # Source file
            rows, cols = data.shape
            src_transform = data.transform
            source = rasterio.band(data, [1, 2, 3])
            # Destination file with new CRS (Rotation)
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
        data.close()
    os.remove(raster_name)
    return([test[0], transform])


def crop_img(img_mat, img_trans, win_corner, win_size):
    """Extract a window with specific characteristics (corner position + size)
    from a rotated snap or timex.

    Parameters
    ----------
    img_mat : np.array
        Array of the rotated image (output of img_rotation function)
    img_trans : tuple
        Transform associated with the rotated image (output of img_rotation function)
    win_corner : tuple
        X,Y coordinates of the bottom-left corner of the extraction window
    win_size : int
        Size of the extraction window in pixels

    Output
    ------
    mat
        Cropped image array from the snap or timex.
    """
    # Define parameters
    coord_w = win_corner
    mean_img = np.mean(img_mat, axis=0)
    shp_img = mean_img.shape

    # Find true coordinates depending on transform parameters
    x = np.arange(img_trans[2],  shp_img[1]*img_trans[0] - np.abs(img_trans[2]), img_trans[0])
    y = np.arange(img_trans[5], (shp_img[0]*img_trans[4] + img_trans[5]), -img_trans[0])

    # Extract window
    x_ind = np.where((x > coord_w[0]) & (x < (coord_w[0]+win_size)))[0]
    y_ind = np.where((y > coord_w[1]) & (y < (coord_w[1]+win_size)))[0]
    mat = mean_img[(y_ind.min()-1):y_ind.max(), (x_ind.min()-1):x_ind.max()]

    return(mat)


def ffill(arr):
    """Fill missing value along the axis 1.

    This function is taken from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    It is mandatory for the bathymetric survey of 06/2021 because it contains a
    lot of missing data near the shore.

    Parameters
    ----------
    arr : np.array
        Array with some NA

    Output
    ------
    out
        Filled array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out
