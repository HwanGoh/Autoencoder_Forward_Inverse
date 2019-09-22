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
    def __init__(self, hyper_p, parameter_dimension, state_dimension, construct_flag):
        
        # Initialize placeholders
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_tf")
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_input_tf")
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_data_tf") # This is needed for batching during training, else can just use state_data
        
        # Initialize weights and biases
        self.layers = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        self.layers[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)
        weights_init_value = 0.05
        biases_init_value = 0       

        if construct_flag == 1:
            with tf.variable_scope("autoencoder") as scope:
                # Forward Problem
                with tf.variable_scope("forward_problem") as scope:
                    for l in range(0, hyper_p.truncation_layer): 
                            W = tf.get_variable("W" + str(l+1), shape = [self.layers[l], self.layers[l + 1]], initializer = tf.contrib.layers.xavier_initializer())
                            b = tf.get_variable("b" + str(l+1), shape = [1, self.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value))                                  
                            tf.summary.histogram("weights" + str(l+1), W)
                            tf.summary.histogram("biases" + str(l+1), b)
                            self.weights.append(W)
                            self.biases.append(b)
                                                 
                # Inverse Problem
                with tf.variable_scope("inverse_problem") as scope:
                    for l in range(hyper_p.truncation_layer, num_layers-1):
                            W = tf.get_variable("W" + str(l+1), shape = [self.layers[l], self.layers[l + 1]], initializer = tf.contrib.layers.xavier_initializer())
                            b = tf.get_variable("b" + str(l+1), shape = [1, self.layers[l + 1]], initializer = tf.constant_initializer(biases_init_value))
                            tf.summary.histogram("weights" + str(l+1), W)
                            tf.summary.histogram("biases" + str(l+1), b)
                            self.weights.append(W)
                            self.biases.append(b)
            
            # Ensures train.Saver only saves the weights and biases                
            self.saver_autoencoder = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "autoencoder")
        
        # Load trained model  
        if construct_flag == 0: 
            graph = tf.get_default_graph()
            for l in range(0, hyper_p.truncation_layer):
                W = graph.get_tensor_by_name("autoencoder/forward_problem/W" + str(l+1) + ':0')
                b = graph.get_tensor_by_name("autoencoder/forward_problem/b" + str(l+1) + ':0')
                self.weights.append(W)
                self.biases.append(b)
            for l in range(hyper_p.truncation_layer, num_layers-1):
                W = graph.get_tensor_by_name("autoencoder/inverse_problem/W" + str(l+1) + ':0')
                b = graph.get_tensor_by_name("autoencoder/inverse_problem/b" + str(l+1) + ':0')
                self.weights.append(W)
                self.biases.append(b)
                
        # Network Propagation
        self.forward_pred = self.forward_problem(self.parameter_input_tf, hyper_p.truncation_layer)
        self.inverse_pred = self.inverse_problem(self.state_input_tf, hyper_p.truncation_layer, len(self.layers))   
        self.autoencoder_pred = self.inverse_problem(self.forward_pred, hyper_p.truncation_layer, len(self.layers)) # To be used in the loss function

    
    def forward_problem(self, X, truncation_layer):  
        with tf.variable_scope("forward_problem") as scope:
            for l in range(0, truncation_layer - 1):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
            W = self.weights[truncation_layer - 1]
            b = self.biases[truncation_layer - 1]
            output = tf.add(tf.matmul(X, W), b)
            return output
    
    def inverse_problem(self, X, truncation_layer, num_layers):   
        with tf.variable_scope("inverse_problem") as scope:
            for l in range(truncation_layer, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
