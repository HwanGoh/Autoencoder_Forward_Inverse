#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:59:12 2019

@author: hwan
"""

from keras.layers import Input, Dense
from keras.models import Model
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class AutoencoderFwdInv:
    def __init__(self, hyper_p, parameter_dimension, state_dimension, construct_flag):        
        # Initialize weights and biases
        self.layers = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        self.layers[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        num_layers = len(self.layers)
        activation = 'tanh'
        activations = ['linear'] + [activation]*hyper_p.num_hidden_layers + [activation]
        activations[hyper_p.truncation_layer] = 'linear' # This is the identity activation
        
        if construct_flag == 1:
        # Forward Problem
            parameter_input = Input(shape=(parameter_dimension,))
            for l in range(1, hyper_p.truncation_layer):
                if l == 1:
                    self.encoded = Dense(self.layers[l], activation=activations[l])(parameter_input)
                else:
                    self.encoded = Dense(self.layers[l], activation=activation[l])(self.encoded)
                               
        # Inverse Problem
            for l in range(hyper_p.truncation_layer, num_layers -1):
                if l == hyper_p.truncation_layer:
                    self.decoded = Dense(self.layers[l], activation=activation)(self.encoded)
                else:
                    self.decoded = Dense(self.layers[l], activation=activation)(self.decoded)

        self.forward_pred = Model(parameter_input, self.encoded)
        self.inverse_pred = Model(self.encoded, self.decoded)
        self.autoencoder_pred = Model(parameter_input, self.decoded)

















