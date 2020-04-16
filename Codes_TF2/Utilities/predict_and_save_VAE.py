#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:11:59 2020

@author: hwan
"""
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def predict_and_save(hyperp, run_options, file_paths, NN, parameter_and_state_obs_test, obs_indices):        
    #######################
    #   Form Predictions  #
    #######################      
    #=== From Parameter Instance ===#
    df_parameter_test = pd.read_csv(file_paths.loadfile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_state_test = pd.read_csv(file_paths.loadfile_name_state_test + '.csv')
    state_test = df_state_test.to_numpy()
            
    #=== Predictions for Reversed Autoencoder ===#
    state_pred = NN.decoder(parameter_test.T)
    if hyperp.data_type == 'bnd':
        state_test_bnd = state_test[obs_indices].flatten()
        state_test_bnd = state_test_bnd.reshape(state_test_bnd.shape[0], 1)
        parameter_pred, _ = NN.encoder(state_test_bnd.T) 
    else:
        parameter_pred, _ = NN.encoder(state_test.T) 
    
    parameter_test = parameter_test.flatten()
    state_test = state_test.flatten()
    parameter_pred = parameter_pred.numpy().flatten()
    state_pred = state_pred.numpy().flatten()
    
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
        
    