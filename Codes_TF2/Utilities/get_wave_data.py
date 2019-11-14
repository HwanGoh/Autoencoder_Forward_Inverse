#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


def load_wave_data(run_options, num_training_data, num_testing_data, batch_size, random_seed):
    
    #=== Load Train and Test Data ===#  
    print('Loading Training Data')
    training_data = scipy.io.loadmat('../../Datasets/DGM_Wave/Samples_TestPrmtrs_F_%s_10Sensors_008FinalTime_%dSamples.mat' %(run_options.computation_domain_discretization, num_training_data))
    parameter_train = training_data['hAS_FEM']
    state_obs_train = np.concatenate((training_data['vxSamplesDataTimeSteps'], training_data['vySamplesDataTimeSteps']), axis = 1)
    
    print('Loading Testing Data')
    testing_data = scipy.io.loadmat('../../Datasets/DGM_Wave/Samples_TestPrmtrs_F_%s_10Sensors_008FinalTime_%dSamples.mat' %(run_options.computation_domain_discretization, num_testing_data))
    parameter_test = testing_data['hAS_FEM']
    state_obs_test = np.concatenate((testing_data['vxSamplesDataTimeSteps'], testing_data['vySamplesDataTimeSteps']), axis = 1)
    
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
    num_training_data = int(0.8 * len(parameter_train))
    parameter_and_state_obs_train = parameter_and_state_obs_train_full.take(num_training_data).batch(batch_size)
    parameter_and_state_obs_val = parameter_and_state_obs_train_full.skip(num_training_data).batch(batch_size)    
    num_batches_train = len(list(parameter_and_state_obs_train))
    num_batches_val = len(list(parameter_and_state_obs_train))

    return parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val