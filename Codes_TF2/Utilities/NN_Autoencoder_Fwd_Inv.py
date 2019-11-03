#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:37:27 2019

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class AutoencoderFwdInv(tf.keras.Model):
    def __init__(self, hyper_p, run_options, parameter_dimension, state_dimension, obs_indices, savefilepath, construct_flag):
        super(AutoencoderFwdInv, self).__init__()
###############################################################################
#                    Constuct Neural Network Architecture                     #
############################################################################### 
        #=== Define Architecture and Create Layer Storage ===#
        self.architecture = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        if run_options.use_full_domain_data == 1 or run_options.use_bnd_data == 1:
            self.architecture[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        if run_options.use_bnd_data_only == 1:
            self.architecture[hyper_p.truncation_layer] = len(obs_indices) # Sets where the forward problem ends and the inverse problem begins
        print(self.architecture)
        self.num_layers = len(self.architecture)    
       
        #=== Define Other Attributes ===#
        self.truncation_layer = hyper_p.truncation_layer
        self.hidden_layers_encoder = [] # This will be a list of layers
        self.hidden_layers_decoder = [] # This will be a list of layers
        activation = ['relu']
        self.activations = ['linear'] + [activation]*hyper_p.num_hidden_layers + ['linear']
        self.activations[hyper_p.truncation_layer] = 'linear' # This is the identity activation
        
        #=== Weights and Biases Initializer ===#
        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.bias_initializer = 'zeros'
                
        if construct_flag == 1:
            self.encoder = Encoder()
            self.decoder = Decoder()

###############################################################################
#                          Autoencoder Propagation                            #    
###############################################################################                
        def call(self, X):
            fwd_problem_solution = self.encoder(X)
            inv_problem_solution = self.decoder(fwd_problem_solution)
            return inv_problem_solution
            
###############################################################################
#                                  Encoder                                    # 
###############################################################################         
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        for l in range(1, self.truncation_layer+1):
            hidden_layer_encoder = tf.keras.layers.Dense(units = self.architecture[l],
                                                         activation = self.activations[l], use_bias = True,
                                                         kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_encoder.append(hidden_layer_encoder)
            
    def call(self, X):
        for hidden_layer in self.hidden_layers_encoder:
            X = hidden_layer(X)
        return 
    
###############################################################################
#                                  Decoder                                    # 
###############################################################################         
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        for l in range(self.truncation_layer+1, self.num_layers):
            hidden_layer_decoder = tf.keras.layers.Dense(units = self.architecture[l],
                                                         activation = self.activations[l], use_bias = True,
                                                         kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_decoder.append(hidden_layer_decoder)
            
    def call(self, X):
        for hidden_layer in self.hidden_layers_decoder:
            X = hidden_layer(X)
        return 
    
    
    
    
    
    
    
    
    
    
    
    
    