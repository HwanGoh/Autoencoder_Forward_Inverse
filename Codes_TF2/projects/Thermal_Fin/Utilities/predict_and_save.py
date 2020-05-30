#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:46:01 2019
@author: hwan
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from Utilities.get_thermal_fin_data import load_thermal_fin_test_data
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def predict_and_save(hyperp, run_options, file_paths):

    ###########################################
    #   Load Data and Trained Neural Network  #
    ###########################################
    #=== Load Testing Data ===#
    obs_indices, parameter_test, state_obs_test, data_input_shape, parameter_dimension\
    = load_thermal_fin_test_data(file_paths, run_options.num_data_test,
            run_options.parameter_dimensions)

    #=== Shuffling Data and Forming Batches ===#
    parameter_and_state_obs_test =\
    tf.data.Dataset.from_tensor_slices((parameter_test, state_obs_test)).shuffle(8192,
            seed=run_options.random_seed).batch(hyperp.batch_size)

    #=== Data and Latent Dimensions of Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        data_dimension = parameter_dimension
        if hyperp.data_type == 'full':
            latent_dimension = run_options.full_domain_dimensions
        if hyperp.data_type == 'bnd':
            latent_dimension = len(obs_indices)

    if run_options.use_reverse_autoencoder == 1:
        if hyperp.data_type == 'full':
            data_dimension = run_options.full_domain_dimensions
        if hyperp.data_type == 'bnd':
            data_dimension = len(obs_indices)
        latent_dimension = parameter_dimension

    #=== Load Trained Neural Network ===#
    NN = AutoencoderFwdInv(hyperp, data_dimension, latent_dimension)
    NN.load_weights(file_paths.NN_savefile_name)

    #######################
    #   Form Predictions  #
    #######################
    #=== From Parameter Instance ===#
    df_parameter_test = pd.read_csv(file_paths.loadfile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_state_test = pd.read_csv(file_paths.loadfile_name_state_test + '.csv')
    state_test = df_state_test.to_numpy()

    #=== Predictions for Standard Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        state_pred = NN.encoder(parameter_test.T)
        if hyperp.data_type == 'bnd':
            state_test_bnd = state_test[obs_indices].flatten()
            state_test_bnd = state_test_bnd.reshape(state_test_bnd.shape[0], 1)
            parameter_pred = NN.decoder(state_test_bnd.T)
        else:
            parameter_pred = NN.decoder(state_test.T)

    #=== Predictions for Reversed Autoencoder ===#
    if run_options.use_reverse_autoencoder == 1:
        state_pred = NN.decoder(parameter_test.T)
        if hyperp.data_type == 'bnd':
            state_test_bnd = state_test[obs_indices].flatten()
            state_test_bnd = state_test_bnd.reshape(state_test_bnd.shape[0], 1)
            parameter_pred = NN.encoder(state_test_bnd.T)
        else:
            parameter_pred = NN.encoder(state_test.T)

    parameter_test = parameter_test.flatten()
    state_test = state_test.flatten()
    parameter_pred = parameter_pred.numpy().flatten()
    state_pred = state_pred.numpy().flatten()

# =============================================================================
#     #=== From Test Batch ===#
#     parameter_and_state_obs_test_draw = parameter_and_state_obs_test.take(1)
#     for batch_num, (parameter_test, state_obs_test) in parameter_and_state_obs_test_draw.enumerate():
#         if run_options.use_standard_autoencoder == 1:
#             parameter_pred_batch = NN.decoder(state_obs_test)
#             state_pred_batch = NN.encoder(parameter_test)
#         if run_options.use_reverse_autoencoder == 1:
#             parameter_pred_batch = NN.encoder(state_obs_test)
#             state_pred_batch = NN.decoder(parameter_test)
#
#     test_index = 6
#     parameter_test = parameter_test[test_index,:].numpy()
#     parameter_pred = parameter_pred_batch[test_index,:].numpy()
#     state_test = state_obs_test[test_index,:].numpy()
#     state_pred = state_pred_batch[test_index,:].numpy()
# =============================================================================

    #####################################
    #   Save Test Case and Predictions  #
    #####################################
    df_parameter_test = pd.DataFrame({'parameter_test': parameter_test})
    df_parameter_test.to_csv(file_paths.savefile_name_parameter_test + '.csv', index=False)
    df_parameter_pred = pd.DataFrame({'parameter_pred': parameter_pred})
    df_parameter_pred.to_csv(file_paths.savefile_name_parameter_pred + '.csv', index=False)
    df_state_test = pd.DataFrame({'state_test': state_test})
    df_state_test.to_csv(file_paths.savefile_name_state_test + '.csv', index=False)
    df_state_pred = pd.DataFrame({'state_pred': state_pred})
    df_state_pred.to_csv(file_paths.savefile_name_state_pred + '.csv', index=False)

    print('\nPredictions Saved to ' + file_paths.NN_savefile_directory)


