#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:37:27 2019

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Autoencoder                                #
###############################################################################
class AutoencoderFwdInv(tf.keras.Model):
    def __init__(self, hyperp, run_options,
                 input_dimensions, latent_dimensions,
                 kernel_initializer, bias_initializer,
                 positivity_constraint):
        super(AutoencoderFwdInv, self).__init__()

        #=== Define Architecture and Create Layer Storage ===#
        self.architecture = [input_dimensions] +\
                [hyperp.num_hidden_nodes_encoder]*hyperp.num_hidden_layers_encoder +\
                [latent_dimensions] +\
                [hyperp.num_hidden_nodes_decoder]*hyperp.num_hidden_layers_decoder +\
                [input_dimensions]

        #=== Define Other Attributes ===#
        self.activations = ['not required'] +\
                [hyperp.activation]*hyperp.num_hidden_layers_encoder +\
                ['linear'] +\
                [hyperp.activation]*hyperp.num_hidden_layers_decoder +\
                ['linear']

        #=== Encoder and Decoder ===#
        self.encoder = Encoder(run_options, positivity_constraint,
                               hyperp.num_hidden_layers_encoder + 1,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer)
        self.decoder = Decoder(run_options, positivity_constraint,
                               hyperp.num_hidden_layers_encoder + 1,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer,
                               len(self.architecture) - 1)

    #=== Autoencoder Propagation ===#
    def call(self, X):
        encoded_input = self.encoder(X)
        decoded_latent = self.decoder(encoded_input)

        return decoded_latent

###############################################################################
#                                  Encoder                                    #
###############################################################################
class Encoder(tf.keras.layers.Layer):
    def __init__(self, run_options,
                 positivity_constraint,
                 truncation_layer, architecture,
                 activations,
                 kernel_initializer, bias_initializer):
        super(Encoder, self).__init__()

        self.run_options = run_options
        self.positivity_constraint = positivity_constraint
        self.truncation_layer = truncation_layer
        self.hidden_layers_encoder = []

        for l in range(1, truncation_layer+1):
            hidden_layer_encoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_encoder.append(hidden_layer_encoder)

    #=== Encoder Propagation ===#
    def call(self, X):
        for hidden_layer in enumerate(self.hidden_layers_encoder):
            if self.run_options.resnet == 1\
                    and 0 < hidden_layer[0] < self.truncation_layer-1:
                X += hidden_layer[1](X)
            else:
                X = hidden_layer[1](X)
        if self.run_options.standard_autoencoder == 1:
            return X
        if self.run_options.reverse_autoencoder == 1:
            return self.positivity_constraint(X)

###############################################################################
#                                  Decoder                                    #
###############################################################################
class Decoder(tf.keras.layers.Layer):
    def __init__(self, run_options,
                 positivity_constraint,
                 truncation_layer, architecture,
                 activations,
                 kernel_initializer, bias_initializer,
                 last_layer_index):
        super(Decoder, self).__init__()

        self.run_options = run_options
        self.positivity_constraint = positivity_constraint
        self.truncation_layer = truncation_layer
        self.last_layer_index = last_layer_index
        self.hidden_layers_decoder = [] # This will be a list of layers

        for l in range(truncation_layer+1, last_layer_index+1):
            hidden_layer_decoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_decoder.append(hidden_layer_decoder)

    #=== Decoder Propagation ===#
    def call(self, X):
        for hidden_layer in enumerate(self.hidden_layers_decoder):
            if self.run_options.resnet == 1\
                    and self.truncation_layer < hidden_layer[0]+self.truncation_layer\
                            < self.last_layer_index-1:
                X += hidden_layer[1](X)
            else:
                X = hidden_layer[1](X)
        if self.run_options.standard_autoencoder == 1:
            return self.positivity_constraint(X)
        if self.run_options.reverse_autoencoder == 1:
            return X
