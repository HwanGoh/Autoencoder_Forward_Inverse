#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:59:12 2019

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.layers import Dropout, Dense
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class AutoencoderFwdInv:
    def __init__(self, hyper_p, parameter_dimension, state_dimension, construct_flag):
        # Initialize placeholders
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_tf")
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_input_tf")
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_data_tf") # This is needed for batching during training, else can just use state_data
        
        # Initialize weights and biases
        self.layers = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        self.layers[hyper_p.truncation_layer-1] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        num_layers = len(self.layers)
        activation = 'tanh'
        
        if construct_flag == 1:
            with tf.variable_scope("autoencoder") as scope:
                # Forward Problem
                with tf.variable_scope("forward_problem") as scope:
                    forward_problem = Sequential()
                    forward_problem.add(Dense(self.layers[1], input_shape=(parameter_dimension,)))
                    for l in range(2, hyper_p.truncation_layer - 2):
                            forward_problem.add(Activation(activation))
                            forward_problem.add(Dense(self.layers[l])
                            
                        else:
                            forward_problem.add(Dense(self.layers[l], activation=activation))
                    forward_problem.add(Dense(self.layers[hyper_p.truncation_layer - 2])
                
                # Inverse Problem
                with tf.variable_scope("inverse_problem") as scope:
                    inverse_problem = Sequential()
                    for l in range(hyper_p.truncation_layer -1, num_layers -1):
                        if l==0:
                            inverse_problem.add(Dense(self.layers[l], activation=activation, input_shape=(parameter_dimension,)))
                        else:
                            inverse_problem.add(Dense(self.layers[l], activation=activation))
                    inverse_problem.add(Dense(self.layers[hyper_p.truncation_layer - 2])



















