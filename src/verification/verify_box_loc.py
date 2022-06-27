#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to verify the position of the box from which we extract the data in the images"""


import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.plot import show
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
from src.data_prep.img_processing import img_rotation, proj_rot

final_df = pd.read_csv('data_CNN/Data_processed/meta_df.csv')
date_unique = final_df['Date'].sort_values().unique()
zm_unique = final_df['Z_m'].sort_values().unique()
bat_df = pd.read_csv("data_CNN/Data_processed/Processed_bathy.csv")


zm = zm_unique[14]
temp_df = final_df[final_df['Z_m'] == zm].reset_index(drop=True)
temp_df['bathy'].unique()
fp_snp = list(temp_df.loc[(temp_df['Type_img'] == 'snap'), 'Fp_img'])
fp_tmx = list(temp_df.loc[(temp_df['Type_img'] == 'timex'), 'Fp_img'])

for i, file in enumerate(fp_snp[:10]):
    data_snp, transform_snp = img_rotation(file)
    #data_tmx, transform_tmx = img_rotation(file)

    # show(np.mean(data_snp, axis=0), transform=transform_snp, cmap='Greys_r')
    # show(np.mean(data_tmx, axis=0), transform=transform_tmx, cmap='Greys_r')

    # Load bathymetric data
    bathy_survey = list(temp_df.loc[(temp_df['Type_img'] == 'snap'), 'bathy'])[i]
    bat = bat_df[['x', 'y', bathy_survey]]
    dst_crs = CRS.from_proj4(proj_rot)
    pts = list(map(lambda x: Point(x), np.array(bat[['x', 'y']])))
    gdf_g = gpd.GeoDataFrame(bat, geometry=pts, crs=dst_crs)

    gdf_g[['x_n', 'y_n']] = gdf_g[['x', 'y']]
    x_arr = np.linspace(np.min(gdf_g['x_n']), np.max(gdf_g['x_n']), 500)
    y_arr = np.linspace(np.min(gdf_g['y_n']), np.max(gdf_g['y_n']), 500)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    # Grid the values
    z_mesh = griddata((gdf_g['x_n'], gdf_g['y_n']), gdf_g[bathy_survey],
                      (x_mesh, y_mesh), method='linear')

    f, ax = plt.subplots(figsize=(6, 6))
    # Plot Image
    show(data_snp, transform=transform_snp, ax=ax)
    # Plot contour
    # CS = ax.contourf(x_mesh, y_mesh, z_mesh, alpha=0.2, cmap='jet',
    # levels=np.arange(-8, 8.5, 0.5))
    CS = ax.contour(x_mesh, y_mesh, z_mesh, alpha=0.6, cmap='jet',
                    levels=np.arange(-8, 8.5, 0.5))
    # Create a Rectangle patch
    rect = patches.Rectangle((70, 70), 256, 256, linewidth=1.5,
                             edgecolor='r', facecolor='none', linestyle='dashed')
    # Add the patch to the Axes
    ax.add_patch(rect)
    #h, v = CS.legend_elements()
    #ax.legend(h, v, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0, 550])
    ax.set_ylim([0, 500])

    f.tight_layout()
