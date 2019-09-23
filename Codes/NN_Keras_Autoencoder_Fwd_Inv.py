#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:59:12 2019

@author: hwan
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class AutoencoderFwdInv:
    def __init__(self, hyper_p, parameter_dimension, state_dimension, construct_flag):        
        # Initialize weights and biases
        self.layers = [parameter_dimension] + [hyper_p.num_hidden_nodes]*hyper_p.num_hidden_layers + [parameter_dimension]
        self.layers[hyper_p.truncation_layer] = state_dimension # Sets where the forward problem ends and the inverse problem begins
        print(self.layers)
        num_layers = len(self.layers)
        activation = 'tanh'
        self.activations = ['linear'] + [activation]*hyper_p.num_hidden_layers + ['linear']
        self.activations[hyper_p.truncation_layer] = 'linear' # This is the identity activation
        
        if construct_flag == 1:
        # Encoder/Forward Problem
            self.parameter_input = Input(shape=(parameter_dimension,), name = 'parameter_input')
            for l in range(1, hyper_p.truncation_layer+1):
                if l == 1:
                    self.encoded = Dense(self.layers[l], activation=self.activations[l], name = 'forward_layer_1')(self.parameter_input)
                else:
                    self.encoded = Dense(self.layers[l], activation=self.activations[l], name = 'forward_layer_' + str(l))(self.encoded)
           
        # Decoder
            for l in range(hyper_p.truncation_layer+1, num_layers):
                if l == hyper_p.truncation_layer+1:
                    self.decoded = Dense(self.layers[l], activation=self.activations[l], name = 'inverse_layer_' + str(l))(self.encoded)
                else:
                    self.decoded = Dense(self.layers[l], activation=self.activations[l], name = 'inverse_layer_' + str(l))(self.decoded)
        
        # Forward and Autoencoder Predictions          
        self.forward_pred = Model(self.parameter_input, self.encoded, name = 'forward_problem')
        self.autoencoder_pred = Model(self.parameter_input, [self.encoded,self.decoded], name = 'autoencoder')
        
        # Inverse Problem (must be defined after full encoder has been defined)    
        self.state_input = Input(shape=(state_dimension,), name = 'state_input')
        for l in range(hyper_p.truncation_layer+1, num_layers):
            if l == hyper_p.truncation_layer+1:
                self.inverse = self.autoencoder_pred.layers[l](self.state_input)
            else:
                self.inverse = self.autoencoder_pred.layers[l](self.inverse)
        
        self.inverse_pred = Model(self.state_input, self.inverse, name = 'inverse_problem')
        
        self.forward_pred.summary()
        self.autoencoder_pred.summary()
        self.inverse_pred.summary()

        pdb.set_trace()














