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
from forward_solve import Fin
from thermal_fin import get_space
from parameter_generator import ParameterGeneratorNineValues
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
#                               Run Options                                   #
###############################################################################
class RunOptions:
    num_hidden_layers = 1
    truncation_layer = 2 # Indexing includes input and output layer
    num_hidden_nodes = 200
    penalty = 10
    num_training_data = 200
    batch_size = 200
    num_epochs = 50000
    gpu    = '3'
    
    filename = f'hlayers{num_hidden_layers}_tlayer{truncation_layer}_hnodes{num_hidden_nodes}_pen{penalty}_data{num_training_data}_batch{batch_size}_epochs{num_epochs}'
    NN_savefile_directory = 'Trained_NNs/' + filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
    NN_savefile_name = NN_savefile_directory + '/' + filename # The file path and name for the four files
    data_savefilepath = 'Data/' + 'data_%d' %(num_training_data)
    
    # Creating Directories
    if not os.path.exists(NN_savefile_directory):
        os.makedirs(NN_savefile_directory)
    
    if not os.path.exists('Data'):
        os.makedirs('Data')
   
###############################################################################
#                                  Driver                                     #
###############################################################################
def trainer(run_options):
    
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following Sheroze's "test_thermal_fin_gradient.py" code
    V = get_space(40)
    solver = Fin(V) 
    
    parameter_true = np.zeros((run_options.num_training_data,V.dim()))
    state_data = np.zeros((run_options.num_training_data,V.dim()))
    
    # Generating Data
    if not os.path.exists('Data'):
        os.makedirs('Data')
        
    if os.path.isfile(run_options.data_savefilepath + '.csv'):
        print('Loading Data')
        df = pd.read_csv(run_options.data_savefilepath + '.csv')
        data = df.to_numpy()
        parameter_true = data[:,0].reshape((run_options.num_training_data,V.dim()))
        state_data = data[:,1].reshape((run_options.num_training_data,V.dim()))
    else:
        for m in range(run_options.num_training_data): 
            print('\nGenerating Parameters and Data Set %d of %d' %(m+1,run_options.num_training_data))
            print(run_options.data_savefilepath)
            # Randomly generate piecewise constant true parameter with 9 values
            parameter_true[m,:], parameter_true_dl = ParameterGeneratorNineValues(V,solver) # True conductivity values       
            # Solve PDE for state variable
            state_data_dl, _ = solver.forward(parameter_true_dl)
            state_data[m,:] = state_data_dl.vector().get_local()           
        # Saving Parameters and State Data
        data = {'parameter_true': parameter_true.flatten(), 'state_data': state_data.flatten()}
        df = pd.DataFrame(data)   
        df.to_csv(run_options.data_savefilepath + '.csv', index=False)  
        
    
    ###########################
    #   Training Properties   #
    ###########################   
    # Neural network
    NN = AutoencoderFwdInv(run_options,parameter_true.shape[1],state_data.shape[1], construct_flag = 1)

    # Loss functional
    with tf.variable_scope('loss') as scope:
        auto_encoder_loss = tf.pow(tf.norm(NN.parameter_input_tf - NN.autoencoder_pred, 2, name= 'auto_encoder_loss'), 2)
        fwd_loss = run_options.penalty*tf.pow(tf.norm(NN.state_data_tf - NN.forward_pred, 2, name= 'fwd_loss'), 2)
        loss = tf.add(auto_encoder_loss, fwd_loss, name="loss")
        tf.summary.scalar("auto_encoder_loss",auto_encoder_loss)
        tf.summary.scalar("fwd_loss",fwd_loss)
        tf.summary.scalar("loss",loss)
                
    # Set optimizers
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001, name = 'adam_opt').minimize(loss)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})            
    # Set GPU configuration options
    gpu_options = tf.GPUOptions(visible_device_list= run_options.gpu,
                                allow_growth=True)
    
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=2,
                            gpu_options= gpu_options)
    
    # Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter('Tensorboard/' + run_options.filename)
    
    ########################
    #   Train Autoencoder  #
    ########################          
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
        
        # Save neural network
        saver = tf.train.Saver()
        saver.save(sess, run_options.NN_savefile_name)
        
        # Train neural network
        print('Beginning Training\n')
        start_time = time.time()
        loss_value = 1000
        num_batches = int(run_options.num_training_data/run_options.batch_size)
        for epoch in range(run_options.num_epochs):
            if num_batches == 1:
                tf_dict = {NN.parameter_input_tf: parameter_true, NN.state_data_tf: state_data} 
                sess.run(optimizer_Adam, tf_dict)   
            else:
                minibatches = random_mini_batches(parameter_true.T, state_data.T, run_options.batch_size, 1234)
                for batch_num in range(num_batches):
                    parameter_true_batch = minibatches[batch_num][0].T
                    state_data_batch = minibatches[batch_num][1].T
                    tf_dict = {NN.parameter_input_tf: parameter_true_batch, NN.state_data_tf: state_data_batch} 
                    sess.run(optimizer_Adam, tf_dict)   
                
            # print to monitor results
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                [loss_value, s] = sess.run([loss,summ], tf_dict)
                writer.add_summary(s,epoch)
                print(run_options.filename)
                print('GPU: ' + run_options.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f\n' %(epoch, loss_value, elapsed))
                start_time = time.time()     
                
            # save every 1000 epochs
            if epoch % 1000 == 0:
                saver.save(sess, run_options.NN_savefile_name, write_meta_graph=False)
        
        # Optimize with LBFGS
        print('Optimizing with LBFGS\n')   
        optimizer_LBFGS.minimize(sess, feed_dict=tf_dict)
        [loss_value, s] = sess.run([loss,summ], tf_dict)
        writer.add_summary(s,run_options.num_epochs)
        
        # Save final model
        saver.save(sess, run_options.NN_savefile_name, write_meta_graph=False)        
    
###############################################################################
#                                   Executor                                  #
###############################################################################     
if __name__ == "__main__":     
    run_options = RunOptions()
    trainer(run_options) 
     
     
     
     
     
     
     
     
     
     
     
     