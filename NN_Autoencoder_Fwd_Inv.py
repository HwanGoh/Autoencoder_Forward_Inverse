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
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None, parameter_dimension])
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None, state_dimension])
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None, state_dimension]) # This is needed for batching during training, else can just use state_data
        
        # Autoencoder
        self.weights, self.biases = self.autoencoder_architecture(parameter_dimension, self.run_options.num_hidden_nodes, state_dimension)
        self.forward_pred = self.forward_problem(self.parameter_input_tf)
        self.autoencoder_pred = self.inverse_problem(self.forward_pred) # To be used in the loss function
        self.inverse_pred = self.inverse_problem(self.state_input_tf)
    
    def autoencoder_architecture(self, parameter_dimension, num_hidden_nodes, state_dimension):
        weights = {
            'encoder_w1': tf.Variable(tf.random_normal([parameter_dimension, num_hidden_nodes])),
            'encoder_w2': tf.Variable(tf.random_normal([num_hidden_nodes, state_dimension])),
            'decoder_w1': tf.Variable(tf.random_normal([state_dimension, num_hidden_nodes])),
            'decoder_w2': tf.Variable(tf.random_normal([num_hidden_nodes, parameter_dimension])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_nodes])),
            'encoder_b2': tf.Variable(tf.random_normal([state_dimension])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_nodes])),
            'decoder_b2': tf.Variable(tf.random_normal([parameter_dimension])),
        }     
        return weights, biases
        
    def forward_problem(self,parameter_input):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(parameter_input, self.weights['encoder_w1']),
                                       self.biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_w2']),
                                       self.biases['encoder_b2']))
        
        return layer_2

    def inverse_problem(self,state_input):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(state_input, self.weights['decoder_w1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_w2']),
                                       self.biases['decoder_b2']))
        return layer_2
     
    def forward_problem_prediction(self, parameter_input):           
        prediction = self.sess.run(self.forward_pred, feed_dict = {self.parameter_input_tf: parameter_input})      
        return prediction
    
    def inverse_problem_prediction(self, state_input):               
        prediction = self.sess.run(self.inverse_pred, feed_dict = {self.state_input_tf: state_input})      
        return prediction