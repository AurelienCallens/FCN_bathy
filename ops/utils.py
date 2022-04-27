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