#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class AutoencoderFwdInv:
    def __init__(self, hyper_p, run_options, parameter_dimension, state_dimension, obs_indices, savefilepath, construct_flag):
        
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################        
        #=== Construct Placeholders ===#
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_tf")
        self.state_obs_tf = tf.placeholder(tf.float32, shape=[None, len(obs_indices)], name = "state_obs_tf") # This is needed for batching during training, else can just use state_data
        
        self.state_obs_inverse_input_tf = tf.placeholder(tf.float32, shape=[None, len(obs_indices)], name = "state_obs_inverse_input_tf")
       
        self.parameter_input_test_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_test_tf")
        self.state_obs_test_tf = tf.placeholder(tf.float32, shape=[None, len(obs_indices)], name = "state_obs_test_tf") # This is needed for batching during training, else can just use state_data
        
        #=== Define Architecture and Create Parameter Storage ===#
        self.layers = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        if run_options.use_full_domain_data == 1 or run_options.use_bnd_data == 1:
            self.layers[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        if run_options.use_bnd_data_only == 1:
            self.layers[hyper_p.truncation_layer] = len(obs_indices) # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        self.weights = [] # This will be a list of tensorflow variables
        self.biases = [] # This will be a list of tensorflow variables
        num_layers = len(self.layers)
        biases_init_value = 0       
        
        #=== Initialize Weights and Biases ===#
        if construct_flag == 1:
            with tf.variable_scope("autoencoder") as scope:
                # Forward Problem
                with tf.variable_scope("encoder") as scope:
                    for l in range(1, hyper_p.truncation_layer+1): 
                            W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.random_normal_initializer())
                            b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(biases_init_value))                                  
                            tf.summary.histogram("weights" + str(l), W)
                            tf.summary.histogram("biases" + str(l), b)
                            self.weights.append(W)
                            self.biases.append(b)
                                                 
                # Inverse Problem
                with tf.variable_scope("decoder") as scope:
                    for l in range(hyper_p.truncation_layer+1, num_layers):
                            W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.contrib.layers.xavier_initializer())
                            b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(biases_init_value))
                            tf.summary.histogram("weights" + str(l), W)
                            tf.summary.histogram("biases" + str(l), b)
                            self.weights.append(W)
                            self.biases.append(b)
        
        #=== Load Trained Model ===# 
        if construct_flag == 0: 
            with tf.variable_scope("encoder") as scope:
                for l in range(1, hyper_p.truncation_layer+1): 
                    df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l) + '.csv')
                    df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l) + '.csv')
                    restored_W = df_trained_weights.values.reshape([self.layers[l-1], self.layers[l]])
                    restored_b = df_trained_biases.values.reshape([1, self.layers[l]])
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.constant_initializer(restored_W))
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(restored_b))                                  
                    self.weights.append(W)
                    self.biases.append(b)
            with tf.variable_scope("decoder") as scope:
               for l in range(hyper_p.truncation_layer+1, num_layers):
                    df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l) + '.csv')
                    df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l) + '.csv')
                    restored_W = df_trained_weights.values.reshape([self.layers[l-1], self.layers[l]])
                    restored_b = df_trained_biases.values.reshape([1, self.layers[l]])
                    W = tf.get_variable("W" + str(l), dtype = tf.float32, shape = [self.layers[l-1], self.layers[l]], initializer = tf.constant_initializer(restored_W))
                    b = tf.get_variable("b" + str(l), dtype = tf.float32, shape = [1, self.layers[l]], initializer = tf.constant_initializer(restored_b))                                  
                    self.weights.append(W)
                    self.biases.append(b)
                
###############################################################################
#                           Network Propagation                               #
###############################################################################                  
        #=== Forward Propagation ===#
        self.encoded = self.encoder(self.parameter_input_tf, hyper_p.truncation_layer)
        self.autoencoder_pred = self.decoder(self.encoded, hyper_p.truncation_layer, len(self.layers)) # To be used in the loss function
        
        self.encoded_test = self.encoder(self.parameter_input_test_tf, hyper_p.truncation_layer)
        self.autoencoder_pred_test = self.decoder(self.encoded_test, hyper_p.truncation_layer, len(self.layers)) # To be used in the loss function
        
        #=== Constructing observed state and inverse problem from observed state ===#
        if run_options.use_bnd_data == 1:
            self.forward_obs_pred = tf.squeeze(tf.gather(self.encoded, obs_indices, axis = 1)) # tf.gather gathers the columns but for some reason it creates a [m,obs_dim,1] tensor that needs to be squeezed
            self.forward_obs_pred_test = tf.squeeze(tf.gather(self.encoded_test, obs_indices, axis = 1))
            self.inverse_pred = self.inverse_problem(self.state_obs_inverse_input_tf, hyper_p.truncation_layer, len(self.layers), obs_indices)  
        if run_options.use_full_domain_data == 1 or run_options.use_bnd_data_only == 1: # Since, for this case, the number of hidden nodes in the truncation layer is equal to the number of sensors, we can just use encoder without gathering for forward prediction and also directly use the decoder for inverse prediction
            self.forward_obs_pred = self.encoded
            self.forward_obs_pred_test = self.encoded_test
            self.inverse_pred = self.decoder(self.state_obs_inverse_input_tf, hyper_p.truncation_layer, len(self.layers))  

###############################################################################
#                                Methods                                      #
############################################################################### 
    def encoder(self, X, truncation_layer):  
        with tf.variable_scope("encoder") as scope:
            for l in range(0, truncation_layer - 1):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
                #tf.summary.histogram("activation" + str(l+1), X)
            W = self.weights[truncation_layer - 1]
            b = self.biases[truncation_layer - 1]
            output = tf.add(tf.matmul(X, W), b)
            return output
    
    def decoder(self, X, truncation_layer, num_layers):   
        with tf.variable_scope("decoder") as scope:
            for l in range(truncation_layer, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
                #tf.summary.histogram("activation" + str(l+1), X)
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
        
    def inverse_problem(self, X, truncation_layer, num_layers, obs_indices): # difference between this and decoder is that the inverse problem utilizes only the observation nodes from the truncation layer and the corresponding weights
        with tf.variable_scope("inverse_problem") as scope:
            self.weights[truncation_layer] = tf.squeeze(tf.gather(self.weights[truncation_layer], obs_indices, axis = 0)) # extract rows of weights corresponding to truncation layer
            for l in range(truncation_layer, num_layers - 2):
                W = self.weights[l]
                b = self.biases[l]
                X = tf.tanh(tf.add(tf.matmul(X, W), b))
            W = self.weights[-1]
            b = self.biases[-1]
            output = tf.add(tf.matmul(X, W), b)
            return output
        
    
