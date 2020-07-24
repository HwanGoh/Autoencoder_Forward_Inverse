#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import numpy as np
import pandas as pd
import time
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 All Data                                    #
###############################################################################
def load_train_and_test_data(file_paths,
        num_training_data, num_testing_data,
        input_dimensions, output_dimensions,
        load_data_train_flag = 1,
        normalize_input_flag = 0, normalize_output_flag = 0):

    start_time_load_data = time.time()

    #=== Load Train and Test Data ===#
    if load_data_train_flag == 1:
        print('Loading Training Data')
        df_input_train = pd.read_csv(file_paths.input_train_savefilepath + '.csv')
        df_output_train = pd.read_csv(file_paths.output_train_savefilepath + '.csv')
        input_train = df_input_train.to_numpy()
        output_train = df_output_train.to_numpy()
        input_train = input_train.reshape((num_training_data,input_dimensions))
        output_train = output_train.reshape((num_training_data,output_dimensions))
    else:
        input_train = []
        output_train = []
    print('Loading Testing Data')
    df_input_test = pd.read_csv(file_paths.input_test_savefilepath + '.csv')
    df_output_test = pd.read_csv(file_paths.output_test_savefilepath + '.csv')
    input_test = df_input_test.to_numpy()
    output_test = df_output_test.to_numpy()
    input_test = input_test.reshape((num_testing_data, input_dimensions))
    output_test = output_test.reshape((num_testing_data, output_dimensions))

    elapsed_time_load_data = time.time() - start_time_load_data
    print('Time taken to load data: %.4f' %(elapsed_time_load_data))

    #=== Normalize Data ===#
    if normalize_input_flag == 1:
        if load_data_train_flag == 1:
            input_train = input_train/\
                    np.expand_dims(np.linalg.norm(input_train, ord = 2, axis = 1), 1)
        input_test = input_test/\
                np.expand_dims(np.linalg.norm(input_test, ord = 2, axis = 1), 1)

    if normalize_output_flag == 1:
        if load_data_train_flag == 1:
            output_train = output_train/\
                    np.expand_dims(np.linalg.norm(output_train, ord = 2, axis = 1), 1)
        output_test = output_test/\
                    np.expand_dims(np.linalg.norm(output_test, ord = 2, axis = 1), 1)

    return input_train, output_train, input_test, output_test
