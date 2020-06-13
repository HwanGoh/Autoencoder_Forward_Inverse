#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:18:47 2019

@author: hwan
"""
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                 Form Training, Validation and Testing Batches               #
###############################################################################
def form_train_val_test_tf_batches(data_train, labels_train, data_test, labels_test, batch_size, random_seed):
    #=== Casting as float32 ===#
    data_train = tf.cast(data_train,tf.float32)
    labels_train = tf.cast(labels_train, tf.float32)
    data_test = tf.cast(data_test, tf.float32)
    labels_test = tf.cast(labels_test, tf.float32)

    #=== Define Input Dimension ===#
    data_input_shape = data_train.shape[-1]

    #=== Dataset Size ===#
    num_data_train = len(data_train)
    num_data_test = len(data_test)

    #=== Shuffling Data ===#
    data_and_labels_train_full = tf.data.Dataset.from_tensor_slices((data_train,
        labels_train)).shuffle(num_data_train, seed=random_seed)
    data_and_labels_test = tf.data.Dataset.from_tensor_slices((data_test,
        labels_test)).batch(batch_size)
    num_batches_test = len(list(data_and_labels_test))

    #=== Partitioning Out Validation Set and Constructing Batches ===#
    current_num_data_train = num_data_train
    num_data_train = int(0.8 * num_data_train)
    num_data_val = current_num_data_train - num_data_train
    data_and_labels_train = data_and_labels_train_full.take(num_data_train).batch(batch_size)
    data_and_labels_val = data_and_labels_train_full.skip(num_data_train).batch(batch_size)
    num_batches_train = len(list(data_and_labels_train))
    num_batches_val = len(list(data_and_labels_val))

    return data_and_labels_train, data_and_labels_val, data_and_labels_test,\
            num_data_train, num_data_val, num_data_test,\
            num_batches_train, num_batches_val, num_batches_test,\
            data_input_shape

###############################################################################
#                      Form Training and Validation Sets                      #
###############################################################################
def form_train_and_val_set(data_train, labels_train):

    #=== Define Input Dimension and Dataset Size ===#
    data_input_shape = data_train.shape[-1]
    num_data_train = data_train.shape[0]
    num_data_split = int(0.8 * num_data_train)

    data_val = data_train[num_data_split:]
    labels_val = labels_train[num_data_split:]
    data_train = data_train[:num_data_split]
    labels_train = labels_train[:num_data_split]

    return data_train, labels_train, data_val, labels_val, data_input_shape
