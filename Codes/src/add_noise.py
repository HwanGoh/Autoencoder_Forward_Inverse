#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def add_noise(options, output_train, output_test, load_data_train_flag = 0):

    np.random.seed(options.random_seed)
    dampening_scalar = 0.001

    #=== Add Noise ===#
    noisy_obs = np.random.choice(
            range(0, output_train.shape[1]), options.num_noisy_obs , replace=False)
    non_noisy_obs = np.setdiff1d(range(0, options.num_obs_points), noisy_obs)
    if load_data_train_flag == 1:
        output_max = np.max(output_train)
        noise = np.random.normal(0, 1, output_train.shape)
        noise[:, non_noisy_obs] = dampening_scalar*noise[:, non_noisy_obs]
        output_train += options.noise_level*output_max*noise
    output_max = np.max(output_test)
    noise = np.random.normal(0, 1, output_test.shape)
    noise[:, non_noisy_obs] = dampening_scalar*noise[:, non_noisy_obs]
    output_test += options.noise_level*output_max*noise

    #=== Noise Regularization Matrix ===#
    diagonal = 1/(options.noise_level*output_max)*np.ones(output_train.shape[1])
    if options.num_noisy_obs_unregularized != 0:
        diagonal[non_noisy_obs[0:options.num_noisy_obs_unregularized]] =\
                (1/dampening_scalar)*\
                diagonal[non_noisy_obs[0:options.num_noisy_obs_unregularized]]
    noise_regularization_matrix = tf.linalg.diag(diagonal)
    noise_regularization_matrix = tf.cast(noise_regularization_matrix, dtype = tf.float32)

    return output_train, output_test, noise_regularization_matrix
