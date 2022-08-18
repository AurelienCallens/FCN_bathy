#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utils function to initialize all the image path before training the networks

Usage:
    from src.utils import initialize_file_path

Author:
    Aurelien Callens - 27/04/2022
"""
import os


def sorted_list_path(dirname, file_ext=".npy"):
    """Create a sorted list of all the files in the indicated repository.

    Parameters
    ----------
    dirname : str
    Filepath of the repository containing the data.
    file_ext : str, optional
    Extension of the files we want to list. The default is ".npy".

    """
    res_list = sorted(
       [
        os.path.join(dirname, fname)
        for fname in os.listdir(dirname)
        if fname.endswith(file_ext)
        ]
    )
    return(res_list)


def initialize_file_path(dir_name, split):
    """
    Create the list of image depending on the indicated split

    Parameters
    ----------
    dir_name : str
        Filepath of the repository containing the data.
    split : str
        Split

    Returns
    -------
    List with two elements: X paths and Y paths
    """
    input_dir = "./data_CNN/" + dir_name + "/" + split + "/Input/"
    target_dir = "./data_CNN/" + dir_name + "/" + split + "/Target/"

    input_img_paths = sorted_list_path(input_dir)
    target_img_paths = sorted_list_path(target_dir)
    return(input_img_paths, target_img_paths)
