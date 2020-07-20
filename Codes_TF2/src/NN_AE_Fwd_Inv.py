#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:37:27 2019

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class AutoencoderFwdInv(tf.keras.Model):
    def __init__(self, hyperp,
            input_dimensions, latent_dimensions,
            kernel_initializer, bias_initializer):
        super(AutoencoderFwdInv, self).__init__()
###############################################################################
#                    Constuct Neural Network Architecture                     #
###############################################################################
        #=== Define Architecture and Create Layer Storage ===#
        self.architecture = [input_dimensions] + [hyperp.num_hidden_nodes]*hyperp.num_hidden_layers +\
                [input_dimensions]
        self.architecture[hyperp.truncation_layer] = latent_dimensions
        print(self.architecture)
        self.num_layers = len(self.architecture)

        #=== Define Other Attributes ===#
        self.hidden_layers_decoder = []
        activation = hyperp.activation
        self.activations = ['not required'] + [activation]*hyperp.num_hidden_layers + ['linear']
        self.activations[hyperp.truncation_layer] = 'linear'
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        #=== Encoder and Decoder ===#
        self.encoder = Encoder(hyperp.truncation_layer,
                               self.architecture, self.activations,
                               self.kernel_initializer, self.bias_initializer)
        self.decoder = Decoder(hyperp.truncation_layer,
                               self.architecture, self.activations,
                               self.kernel_initializer, self.bias_initializer,
                               self.num_layers)

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
    def __init__(self, truncation_layer, architecture, activations,
            kernel_initializer, bias_initializer):
        super(Encoder, self).__init__()
        self.hidden_layers_encoder = []
        for l in range(1, truncation_layer+1):
            hidden_layer_encoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_encoder.append(hidden_layer_encoder)
    def call(self, X):
        for hidden_layer in self.hidden_layers_encoder:
            X = hidden_layer(X)
        return X

###############################################################################
#                                  Decoder                                    #
###############################################################################
class Decoder(tf.keras.layers.Layer):
    def __init__(self, truncation_layer, architecture, activations,
            kernel_initializer, bias_initializer,
            num_layers):
        super(Decoder, self).__init__()
        self.hidden_layers_decoder = [] # This will be a list of layers
        for l in range(truncation_layer+1, num_layers):
            hidden_layer_decoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_decoder.append(hidden_layer_decoder)
    def call(self, X):
        for hidden_layer in self.hidden_layers_decoder:
            X = hidden_layer(X)
        return X












