#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:37:53 2022

@author: aurelien
"""

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Name of config file (must be placed in ./configs/")
    args = parser.parse_args()

    params_file = 'configs/' + args.config

    print(params_file)


