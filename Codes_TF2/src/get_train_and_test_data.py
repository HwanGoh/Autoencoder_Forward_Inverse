#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import tensorflow as tf
import pandas as pd
import time
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 All Data                                    #
###############################################################################
def load_train_and_test_data(file_paths, num_training_data, num_testing_data,
        data_dimensions, label_dimensions, load_data_train_flag = 1):

    start_time_load_data = time.time()

    #=== Load Train and Test Data ===#
    if load_data_train_flag == 1:
        print('Loading Training Data')
        df_data_train = pd.read_csv(file_paths.data_train_savefilepath + '.csv')
        df_labels_train = pd.read_csv(file_paths.labels_train_savefilepath + '.csv')
        data_train = df_data_train.to_numpy()
        labels_train = df_labels_train.to_numpy()
        data_train = data_train.reshape((num_training_data,data_dimensions))
        labels_train = labels_train.reshape((num_training_data,label_dimensions))
    else:
        data_train = []
        labels_train = []
    print('Loading Testing Data')
    df_data_test = pd.read_csv(file_paths.data_test_savefilepath + '.csv')
    df_labels_test = pd.read_csv(file_paths.labels_test_savefilepath + '.csv')
    data_test = df_data_test.to_numpy()
    labels_test = df_labels_test.to_numpy()
    data_test = data_test.reshape((num_testing_data, data_dimensions))
    labels_test = labels_test.reshape((num_testing_data, label_dimensions))

    elapsed_time_load_data = time.time() - start_time_load_data
    print('Time taken to load data: %.4f' %(elapsed_time_load_data))

    return data_train, labels_train, data_test, labels_test
