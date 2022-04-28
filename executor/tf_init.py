#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 08:42:59 2022

@author: aurelien
"""

import tensorflow as tf

def start_tf_session(mode, cpu_core=2):
    if mode == 'cpu' and type(cpu_core) == int:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_core)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_core)
        tf.config.set_soft_device_placement(True)
    else:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.80
        tf.compat.v1.Session(config=config)

