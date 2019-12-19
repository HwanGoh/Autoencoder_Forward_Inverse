#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:46:01 2019

@author: hwan
"""
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def predict_and_save(hyperp, run_options, file_paths, NN, parameter_and_state_obs_test, obs_indices):        
    #######################
    #   Form Predictions  #
    #######################      
    #=== From Parameter Instance ===#
    df_parameter_test = pd.read_csv(file_paths.savefile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_state_test = pd.read_csv(file_paths.savefile_name_state_test + '.csv')
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
    
    #=== From Test Batch ===#
# =============================================================================
#     parameter_and_state_obs_test_draw = parameter_and_state_obs_test.take(1)
#     for batch_num, (parameter_test, state_obs_test) in parameter_and_state_obs_test_draw.enumerate():
#         if run_options.use_standard_autoencoder == 1:
#             parameter_pred_batch = NN.decoder(state_obs_test)
#             state_pred_batch = NN.encoder(parameter_test)
#         if run_options.use_reversed_autoencoder == 1:
#             parameter_pred_batch = NN.encoder(state_obs_test)
#             state_pred_batch = NN.decoder(parameter_test)
#           
#     parameter_test = parameter_test[4,:].numpy()
#     parameter_pred = parameter_pred_batch[4,:].numpy()
#     state_test = state_obs_test[4,:].numpy()
#     state_pred = state_pred_batch[4,:].numpy()
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
        
    