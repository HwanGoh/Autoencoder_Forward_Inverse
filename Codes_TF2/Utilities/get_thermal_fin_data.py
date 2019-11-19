#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 All Data                                    #
###############################################################################
def load_thermal_fin_data(run_options, num_training_data, batch_size, random_seed):
    
    #=== Load observation indices ===# 
    print('Loading Boundary Indices')
    obs_indices = np.loadtxt(open(run_options.observation_indices_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)

    #=== Load Train and Test Data ===#  
    print('Loading Training Data')
    parameter_train = np.loadtxt(open(run_options.parameter_train_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
    state_obs_train = np.loadtxt(open(run_options.state_obs_train_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
    parameter_train = parameter_train.reshape((num_training_data, run_options.parameter_dimensions))
    state_obs_train = state_obs_train.reshape((num_training_data, len(obs_indices)))
    print('Loading Testing Data')
    parameter_test = np.loadtxt(open(run_options.parameter_test_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
    state_obs_test = np.loadtxt(open(run_options.state_obs_test_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
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
    
    #=== Shuffling Data ===#
    parameter_and_state_obs_train_full = tf.data.Dataset.from_tensor_slices((parameter_train, state_obs_train)).shuffle(8192, seed=random_seed)
    parameter_and_state_obs_test = tf.data.Dataset.from_tensor_slices((parameter_test, state_obs_test)).shuffle(8192, seed=random_seed).batch(batch_size)
    
    #=== Partitioning Out Validation Set and Constructing Batches ===#
    num_training_data = int(0.8 * num_training_data)
    parameter_and_state_obs_train = parameter_and_state_obs_train_full.take(num_training_data).batch(batch_size)
    parameter_and_state_obs_val = parameter_and_state_obs_train_full.skip(num_training_data).batch(batch_size)    
    num_batches_train = len(list(parameter_and_state_obs_train))
    num_batches_val = len(list(parameter_and_state_obs_train))

    return obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val

###############################################################################
#                                 Test Data                                   #
###############################################################################
def load_thermal_fin_test_data(run_options, batch_size, random_seed):
    
    #=== Load observation indices ===# 
    print('Loading Boundary Indices')
    obs_indices = np.loadtxt(open(run_options.observation_indices_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)

    print('Loading Testing Data')
    parameter_test = np.loadtxt(open(run_options.parameter_test_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
    state_obs_test = np.loadtxt(open(run_options.state_obs_test_savefilepath + '.csv', "rb"), delimiter=",", skiprows=1)
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