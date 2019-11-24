#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:18:47 2019

@author: hwan
"""
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def form_train_val_test_batches(data_train, labels_train, data_test, labels_test, batch_size, random_seed):
    num_training_data = len(data_train)
    num_testing_data = len(data_test)
    
    #=== Shuffling Data ===#
    data_and_labels_train_full = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).shuffle(num_training_data, seed=random_seed)
    data_and_labels_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).batch(batch_size)

    #=== Partitioning Out Validation Set and Constructing Batches ===#
    num_training_data = int(0.8 * num_training_data)
    data_and_labels_train = data_and_labels_train_full.take(num_training_data).batch(batch_size)
    data_and_labels_val = data_and_labels_train_full.skip(num_training_data).batch(batch_size)    
    num_batches_train = len(list(data_and_labels_train))
    num_batches_val = len(list(data_and_labels_val))

    return data_and_labels_train, data_and_labels_val, data_and_labels_test, num_training_data, num_testing_data, num_batches_train, num_batches_val
