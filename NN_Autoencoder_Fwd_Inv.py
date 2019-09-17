#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class AutoencoderFwdInv:
    def __init__(self, run_options, parameter_dimension, state_dimension):
        self.run_options =  run_options      
              
        # Initialize placeholders
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension], name = "parameter_input_tf")
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_input_tf")
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None, state_dimension], name = "state_data_tf") # This is needed for batching during training, else can just use state_data
        
        # Autoencoder
        with tf.name_scope('NN_Feed_Forward') as scope:
            self.weights, self.biases = self.autoencoder_architecture(parameter_dimension, self.run_options.num_hidden_nodes, state_dimension)
            self.forward_pred = self.forward_problem(self.parameter_input_tf)
            self.autoencoder_pred = self.inverse_problem(self.forward_pred) # To be used in the loss function
            self.inverse_pred = self.inverse_problem(self.state_input_tf)
    
    def autoencoder_architecture(self, parameter_dimension, num_hidden_nodes, state_dimension):
        weights = {
            'encoder_w1': tf.Variable(tf.random_normal([parameter_dimension, num_hidden_nodes]), name = "encoder_w1"),
            'encoder_w2': tf.Variable(tf.random_normal([num_hidden_nodes, state_dimension]), name = "encoder_w2"),
            'decoder_w1': tf.Variable(tf.random_normal([state_dimension, num_hidden_nodes]), name = "decoder_w1"),
            'decoder_w2': tf.Variable(tf.random_normal([num_hidden_nodes, parameter_dimension]), name = "decoder_w2"),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_nodes]), name = "encoder_b1"),
            'encoder_b2': tf.Variable(tf.random_normal([state_dimension]), name = "encoder_b2"),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_nodes]), name = "decoder_b1"),
            'decoder_b2': tf.Variable(tf.random_normal([parameter_dimension]), name = "decoder_b2"),
        }     
        return weights, biases
        
    def forward_problem(self,parameter_input):
        with tf.name_scope('Fwd_Problem') as scope:
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(parameter_input, self.weights['encoder_w1']),
                                           self.biases['encoder_b1']), name = "Fwd_Layer_1")
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_w2']),
                                           self.biases['encoder_b2']), name = "Fwd_Layer_1")
            
            return layer_2

    def inverse_problem(self,state_input):
        with tf.name_scope('Inv_Problem') as scope:
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(state_input, self.weights['decoder_w1']),
                                           self.biases['decoder_b1']), name = "Inv_Layer_1")
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_w2']),
                                           self.biases['decoder_b2']), name = "Inv_Layer_1")
            return layer_2