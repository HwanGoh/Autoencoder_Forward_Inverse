#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:29:36 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import time
import os

from random_mini_batches import random_mini_batches

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class AutoencoderFwdInv:
    def __init__(self, run_options, parameter_true, state_data):
        
        self.run_options =  run_options
        
        # Initialize placeholders
        self.parameter_input_tf = tf.placeholder(tf.float32, shape=[None,parameter_true.shape[1]])
        self.state_input_tf = tf.placeholder(tf.float32, shape=[None,state_data.shape[1]])
        self.state_data_tf = tf.placeholder(tf.float32, shape=[None,state_data.shape[1]]) # This is needed for batching during training, else can just use state_data
        
        # Autoencoder
        self.weights, self.biases = self.autoencoder_architecture(parameter_true.shape[1], self.run_options.num_hidden_nodes, state_data.shape[1])
        self.forward_pred = self.forward_problem(self.parameter_input_tf)
        self.autoencoder_pred = self.inverse_problem(self.forward_pred) # To be used in the loss function
        self.inverse_pred = self.inverse_problem(self.state_input_tf)
        
        # Loss functional
        self.loss = tf.pow(tf.norm(self.parameter_input_tf - self.autoencoder_pred, 2), 2) + \
                    self.run_options.penalty*tf.pow(tf.norm(self.state_data_tf - self.forward_pred, 2), 2)
                    
        # Set Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.lbfgs = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                             method='L-BFGS-B',
                                                             options={'maxiter':10000,
                                                                      'maxfun':50000,
                                                                      'maxcor':50,
                                                                      'maxls':50,
                                                                      'ftol':1.0 * np.finfo(float).eps})            
        # Set GPU configuration options
        self.gpu_options = tf.GPUOptions(visible_device_list= self.run_options.gpu,
                                         allow_growth=True)
        
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     intra_op_parallelism_threads=4,
                                     inter_op_parallelism_threads=2,
                                     gpu_options=self.gpu_options)
        
        
        # Tensorflow Session
        self.sess = tf.Session(config=self.config)       
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # Save neural network
        if not os.path.exists(self.run_options.savefilepath):
            os.makedirs(self.run_options.savefilepath)
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.run_options.savefilename, write_meta_graph=False)
        
        # Train neural network
        self.train(parameter_true, state_data)  
    
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
    
    def train(self, parameter_true, state_data):     
        self.saver.save(self.sess, self.run_options.savefilename) # preparing to save session
        print('Beginning Training\n')
        start_time = time.time()
        loss_value = 1000
        for epoch in range(self.run_options.num_epochs): 
            minibatches = random_mini_batches(parameter_true.T, state_data.T, self.run_options.batch_size, 1234)
            for batch_num in range(self.run_options.num_batches):
                parameter_true_batch = minibatches[batch_num][0].T
                state_data_batch = minibatches[batch_num][1].T
                tf_dict = {self.parameter_input_tf: parameter_true_batch, self.state_data_tf: state_data_batch} 
                self.sess.run(self.train_op_Adam, tf_dict)     
                
            # print to monitor results
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print(self.run_options.filename)
                print('GPU: ' + self.run_options.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f\n' %(epoch, loss_value, elapsed))
                start_time = time.time()     
                
            # save every 1000 epochs
            if epoch % 1000 == 0:
                self.saver.save(self.sess, self.run_options.savefilename, write_meta_graph=False)

    def predict(self, X_star):        
        tf_dict = {self.x_data_tf: X_star[:, 0:1], self.t_data_tf: X_star[:, 1:2],
                   self.x_phys_tf: X_star[:, 0:1], self.t_phys_tf: X_star[:, 1:2]}        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)        
        return u_star, f_star

    def forward_problem_prediction(self, parameter_input):           
        prediction = self.sess.run(self.forward_pred, feed_dict = {self.parameter_input_tf: parameter_input})      
        return prediction
    
    def inverse_problem_prediction(self, state_input):               
        prediction = self.sess.run(self.inverse_pred, feed_dict = {self.state_input_tf: state_input})      
        return prediction