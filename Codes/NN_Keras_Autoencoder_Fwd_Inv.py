#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:59:12 2019

@author: hwan
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
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
        self.layers[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        num_layers = len(self.layers)
        activation = 'tanh'
        activations = ['linear'] + [activation]*hyper_p.num_hidden_layers + [activation]
        activations[hyper_p.truncation_layer] = 'linear'
        
        pdb.set_trace()
        if construct_flag == 1:
            with tf.variable_scope("autoencoder") as scope:
                # Forward Problem
                with tf.variable_scope("forward_problem") as scope:
                    parameter_input = Input(shape=(parameter_dimension,))
                    for l in range(1, hyper_p.truncation_layer):
                        if l == 1:
                            encoded = Dense(self.layers[l], activation=activations[l])(parameter_input)
                        else:
                            encoded = Dense(self.layers[l], activation=activation[l])(encoded)
                                       
                # Inverse Problem
                with tf.variable_scope("inverse_problem") as scope:
                    for l in range(hyper_p.truncation_layer, num_layers -1):
                        if l == hyper_p.truncation_layer:
                            decoded = Dense(self.layers[l], activation=activation)(encoded)
                        else:
                            decoded = Dense(self.layers[l], activation=activation)(decoded)




















