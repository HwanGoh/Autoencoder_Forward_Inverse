#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import tensorflow as tf
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 All Data                                    #
###############################################################################
def load_thermal_fin_data(run_options, num_training_data):
    
    #=== Load observation indices ===# 
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(run_options.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 

    #=== Load Train and Test Data ===#  
    print('Loading Training Data')
    df_parameter_train = pd.read_csv(run_options.parameter_train_savefilepath + '.csv')
    df_state_obs_train = pd.read_csv(run_options.state_obs_train_savefilepath + '.csv')
    parameter_train = df_parameter_train.to_numpy()
    state_obs_train = df_state_obs_train.to_numpy()
    parameter_train = parameter_train.reshape((num_training_data, run_options.parameter_dimensions))
    state_obs_train = state_obs_train.reshape((num_training_data, len(obs_indices)))
    print('Loading Testing Data')
    df_parameter_test = pd.read_csv(run_options.parameter_test_savefilepath + '.csv')
    df_state_obs_test = pd.read_csv(run_options.state_obs_test_savefilepath + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    state_obs_test = df_state_obs_test.to_numpy()
    parameter_test = parameter_test.reshape((run_options.num_testing_data, run_options.parameter_dimensions))
    state_obs_test = state_obs_test.reshape((run_options.num_testing_data, len(obs_indices)))

    #=== Casting as float32 ===#
    parameter_train = tf.cast(parameter_train,tf.float32)
    state_obs_train = tf.cast(state_obs_train, tf.float32)
    parameter_test = tf.cast(parameter_test, tf.float32)
    state_obs_test = tf.cast(state_obs_test, tf.float32)
        
    #=== Define Outputs ===#
    data_input_shape = parameter_train.shape[1:]
    parameter_dimension = parameter_train.shape[-1]

    return obs_indices, parameter_train, state_obs_train, parameter_test, state_obs_test, data_input_shape, parameter_dimension

###############################################################################
#                                 Test Data                                   #
###############################################################################
def load_thermal_fin_test_data(run_options, batch_size, random_seed):
    
    #=== Load observation indices ===# 
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(run_options.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 

    print('Loading Testing Data')
    df_parameter_test = pd.read_csv(run_options.parameter_test_savefilepath + '.csv')
    df_state_obs_test = pd.read_csv(run_options.state_obs_test_savefilepath + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    state_obs_test = df_state_obs_test.to_numpy()
    parameter_test = parameter_test.reshape((run_options.num_testing_data, run_options.parameter_dimensions))
    state_obs_test = state_obs_test.reshape((run_options.num_testing_data, len(obs_indices)))

    #=== Casting as float32 ===#
    parameter_test = tf.cast(parameter_test, tf.float32)
    state_obs_test = tf.cast(state_obs_test, tf.float32)
        
    #=== Define Outputs ===#
    data_input_shape = parameter_test.shape[1:]
    parameter_dimension = parameter_test.shape[-1]
    
    #=== Shuffling Data ===#
    parameter_and_state_obs_test = tf.data.Dataset.from_tensor_slices((parameter_test, state_obs_test)).shuffle(8192, seed=random_seed).batch(batch_size)
    
    return obs_indices, parameter_and_state_obs_test, data_input_shape, parameter_dimension