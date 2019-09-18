#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class AutoencoderFwdInv:
    def __init__(self, run_options, parameter_dimension, state_dimension):
        
        # Initialize placeholders
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_tf")
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_input_tf")
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_data_tf") # This is needed for batching during training, else can just use state_data
        
        # Initialize weights and biases
        self.layers = [parameter_dimension] + [run_options.num_hidden_nodes]*run_options.num_hidden_layers + [parameter_dimension]
        self.layers[run_options.truncation_layer-1] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            self.weights.append(W)
            self.biases.append(b)
        
        # Network Propagation
        self.forward_pred = self.forward_problem(self.parameter_input_tf, run_options.truncation_layer)
        self.autoencoder_pred = self.inverse_problem(self.forward_pred, run_options.truncation_layer, len(self.layers)) # To be used in the loss function
        self.inverse_pred = self.inverse_problem(self.state_input_tf, run_options.truncation_layer, len(self.layers))
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def forward_problem(self, X, truncation_layer):       
        for l in range(0, truncation_layer - 2):
            W = self.weights[l]
            b = self.biases[l]
            X = tf.tanh(tf.add(tf.matmul(X, W), b))
        W = self.weights[truncation_layer - 2]
        b = self.biases[truncation_layer - 2]
        output = tf.add(tf.matmul(X, W), b)
        return output
    
    def inverse_problem(self, X, truncation_layer, num_layers):       
        for l in range(truncation_layer-1, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            X = tf.tanh(tf.add(tf.matmul(X, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        output = tf.add(tf.matmul(X, W), b)
        return output
