#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:00:39 2022

@author: aurelien
"""
import os


def sorted_list_path(dirname, file_ext=".npy"):
   res_list = sorted(
       [
           os.path.join(dirname, fname)
           for fname in os.listdir(dirname)
           if fname.endswith(file_ext)
       ]
   )
   return(res_list)


def initialize_file_path(dir_name, split):

    input_dir = "./data_CNN/" + dir_name + "/" + split + "/Input/"
    target_dir = "./data_CNN/" + dir_name + "/" + split + "/Target/"

    input_img_paths = sorted_list_path(input_dir)
    target_img_paths = sorted_list_path(target_dir)
    return(input_img_paths, target_img_paths)
