#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""
import sys
sys.path.append('../')

from forward_solve import Fin
from thermal_fin import get_space
from parameter_generator import ParameterGeneratorNineValues

import numpy as np
import tensorflow as tf
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
#                               Parameters                                    #
###############################################################################
class RunOptions:
    num_hidden_nodes = 200
    penalty = 1
    num_training_data = 10
    batch_size = 4
    num_batches = int(num_training_data/batch_size)
    num_epochs = 1000
    gpu    = '0'
    
    filename = f'hnodes{num_hidden_nodes}_pen{penalty}_data{num_training_data}_batch{batch_size}_epochs{num_epochs}'
    savefilepath = 'Models/' + filename
    savefilename = savefilepath + '/' + filename
   
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    run_options = RunOptions()
    
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following "test_thermal_fin_gradient.py" code
    V = get_space(40)
    solver = Fin(V) 
    
    parameter_true = np.zeros((run_options.num_training_data,1446))
    state_data = np.zeros((run_options.num_training_data,1446))
    
    for m in range(run_options.num_training_data): 
        print(run_options.filename[:-3])
        # Randomly generate piecewise constant true parameter with 9 values
        parameter_true[m,:], parameter_true_dl = ParameterGeneratorNineValues(V,solver) # True conductivity values       
        # Solve PDE for state variable
        state_data_dl,_, _, _,_ = solver.forward(parameter_true_dl)
        state_data[m,:] = state_data_dl.vector().get_local()
    
    ###########################
    #   Training Properties   #
    ###########################   
    # Neural network
    NN = AutoencoderFwdInv(run_options,parameter_true.shape[1],state_data.shape[1])
    
    # Loss functional
    loss = tf.pow(tf.norm(NN.parameter_input_tf - NN.autoencoder_pred, 2), 2) + \
           NN.run_options.penalty*tf.pow(tf.norm(NN.state_data_tf - NN.forward_pred, 2), 2)
                
    # Set optimizers
    optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op_Adam = optimizer_Adam.minimize(loss)
    lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                   method='L-BFGS-B',
                                                   options={'maxiter':10000,
                                                            'maxfun':50000,
                                                            'maxcor':50,
                                                            'maxls':50,
                                                            'ftol':1.0 * np.finfo(float).eps})            
    # Set GPU configuration options
    gpu_options = tf.GPUOptions(visible_device_list= run_options.gpu,
                                allow_growth=True)
    
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=2,
                            gpu_options= gpu_options)

    ########################
    #   Train Autoencoder  #
    ########################          
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) 
        
        # Save neural network
        if not os.path.exists(run_options.savefilepath):
            os.makedirs(run_options.savefilepath)
        saver = tf.train.Saver()
        saver.save(sess, run_options.savefilename)
        
        # Train neural network
        print('Beginning Training\n')
        start_time = time.time()
        loss_value = 1000
        for epoch in range(run_options.num_epochs): 
            minibatches = random_mini_batches(parameter_true.T, state_data.T, run_options.batch_size, 1234)
            for batch_num in range(run_options.num_batches):
                parameter_true_batch = minibatches[batch_num][0].T
                state_data_batch = minibatches[batch_num][1].T
                tf_dict = {NN.parameter_input_tf: parameter_true_batch, NN.state_data_tf: state_data_batch} 
                sess.run(train_op_Adam, tf_dict)     
                
            # print to monitor results
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = sess.run(loss, tf_dict)
                print(run_options.filename)
                print('GPU: ' + run_options.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f\n' %(epoch, loss_value, elapsed))
                start_time = time.time()     
                
            # save every 1000 epochs
            if epoch % 1000 == 0:
                saver.save(sess, run_options.savefilename, write_meta_graph=False)
                
    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     