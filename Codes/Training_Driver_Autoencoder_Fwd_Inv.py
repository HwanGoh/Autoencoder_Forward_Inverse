#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""


import sys
sys.path.append('../')

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
import numpy as np
import pandas as pd

from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from random_mini_batches import random_mini_batches
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                       Hyperparameters and Filenames                         #
###############################################################################
class HyperParameters:
    num_hidden_layers = 1
    truncation_layer  = 2 # Indexing includes input and output layer
    num_hidden_nodes  = 200
    penalty           = 10
    num_training_data = 20
    batch_size        = 20
    num_epochs        = 2000
    gpu               = '0'
    
class FileNames:
    def __init__(self,hyper_p):        
        self.filename = f'hl{hyper_p.num_hidden_layers}_tl{hyper_p.truncation_layer}_hn{hyper_p.num_hidden_nodes}_p{hyper_p.penalty}_d{hyper_p.num_training_data}_b{hyper_p.batch_size}_e{hyper_p.num_epochs}'
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.data_savefilepath = '../Data/' + 'data_%d' %(hyper_p.num_training_data)
        
        # Creating Directories
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)
   
###############################################################################
#                                  Driver                                     #
###############################################################################
def trainer(hyper_p, filenames):
    
    hyper_p.batch_size = hyper_p.num_training_data
    
    # Loading Data        
    if os.path.isfile(filenames.data_savefilepath + '.csv'):
        print('Loading Data')
        df = pd.read_csv(filenames.data_savefilepath + '.csv')
        true_data = df.to_numpy()
        parameter_data = true_data[:,0].reshape((hyper_p.num_training_data, 1446))
        state_data = true_data[:,1].reshape((hyper_p.num_training_data, 1446))
    else:
        raise ValueError('Data of size %d has not yet been generated' %(hyper_p.num_training_data))
        
    ###########################
    #   Training Properties   #
    ###########################   
    # Neural network
    NN = AutoencoderFwdInv(hyper_p,parameter_data.shape[1],state_data.shape[1], construct_flag = 1)
    
    # Loss functional
    with tf.variable_scope('loss') as scope:
        auto_encoder_loss = tf.pow(tf.norm(NN.parameter_input_tf - NN.autoencoder_pred, 2, name= 'auto_encoder_loss'), 2)
        fwd_loss = hyper_p.penalty*tf.pow(tf.norm(NN.state_data_tf - NN.forward_pred, 2, name= 'fwd_loss'), 2)
        loss = tf.add(auto_encoder_loss, fwd_loss, name="loss")
        tf.summary.scalar("auto_encoder_loss",auto_encoder_loss)
        tf.summary.scalar("fwd_loss",fwd_loss)
        tf.summary.scalar("loss",loss)
                
    # Set optimizers
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})            
    # Set GPU configuration options
    gpu_options = tf.GPUOptions(visible_device_list=hyper_p.gpu,
                                allow_growth=True)
    
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=2,
                                gpu_options= gpu_options)
    
    # Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter('../Tensorboard/' + filenames.filename)
    
    ########################
    #   Train Autoencoder  #
    ########################          
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
        
        # Save neural network
        saver = tf.train.Saver(NN.saver_autoencoder)
        saver.save(sess, filenames.NN_savefile_name)
        
        # Train neural network
        print('Beginning Training\n')
        start_time = time.time()
        num_batches = int(hyper_p.num_training_data/hyper_p.batch_size)
        for epoch in range(hyper_p.num_epochs):
            if num_batches == 1:
                tf_dict = {NN.parameter_input_tf: parameter_data, NN.state_data_tf: state_data} 
                sess.run(optimizer_Adam, tf_dict)   
            else:
                minibatches = random_mini_batches(parameter_data.T, state_data.T, hyper_p.batch_size, 1234)
                for batch_num in range(num_batches):
                    parameter_data_batch = minibatches[batch_num][0].T
                    state_data_batch = minibatches[batch_num][1].T
                    tf_dict = {NN.parameter_input_tf: parameter_data_batch, NN.state_data_tf: state_data_batch} 
                    sess.run(optimizer_Adam, tf_dict)   
                
            # print to monitor results
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                [loss_value, s] = sess.run([loss,summ], tf_dict)
                writer.add_summary(s,epoch)
                print(filenames.filename)
                print('GPU: ' + hyper_p.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f\n' %(epoch, loss_value, elapsed))
                start_time = time.time()     
                
            # save every 1000 epochs
            if epoch % 1000 == 0:
                saver.save(sess, filenames.NN_savefile_name, write_meta_graph=False)
                
        
        # Optimize with LBFGS
        print('Optimizing with LBFGS\n')   
        optimizer_LBFGS.minimize(sess, feed_dict=tf_dict)
        [loss_value, s] = sess.run([loss,summ], tf_dict)
        writer.add_summary(s,hyper_p.num_epochs)
        print('LBFGS Optimization Complete\n') 
        elapsed = time.time() - start_time
        print('Loss: %.3e, Time: %.2f\n' %(loss_value, elapsed))
        
        # Save final model
        saver.save(sess, filenames.NN_savefile_name, write_meta_graph=False)   
        print('Final Model Saved')  
    
###############################################################################
#                                   Executor                                  #
###############################################################################     
if __name__ == "__main__":     
    hyper_p = HyperParameters()
    if len(sys.argv) > 1:
            hyper_p.num_hidden_layers = int(sys.argv[1])
            hyper_p.truncation_layer  = int(sys.argv[2])
            hyper_p.num_hidden_nodes  = int(sys.argv[3])
            hyper_p.penalty           = int(sys.argv[4])
            hyper_p.num_training_data = int(sys.argv[5])
            hyper_p.batch_size        = int(sys.argv[6])
            hyper_p.num_epochs        = int(sys.argv[7])
            hyper_p.gpu               = str(sys.argv[8])
        
    filenames = FileNames(hyper_p)
    trainer(hyper_p, filenames) 
    
     
     
     
     
     
     
     
     
     
     
     
     