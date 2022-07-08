#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Function to initialize tensorflow session with CPU or GPU

Usage:
    from src.executor.tf_init import start_tf_session

Author:
    Aurelien Callens - 14/04/2022
"""
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def start_tf_session(mode, cpu_core=0):
    """Start tensorflow session with CPU or GPU.

    Parameters
    ----------
    mode: str
        'cpu' or 'gpu'
    """
    if mode == 'cpu' and type(cpu_core) == int:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_core)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.set_soft_device_placement(True)
    else:
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

