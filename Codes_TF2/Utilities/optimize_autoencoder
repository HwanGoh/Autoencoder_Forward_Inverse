#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:31:44 2019

@author: hwan
"""
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL) # Suppresses all the messages when run begins
import numpy as np

import shutil # for deleting directories
import os
import time

from random_mini_batches import random_mini_batches
from compute_batch_metric import compute_batch_metric
from save_trained_parameters import save_weights_and_biases

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def optimize_autoencoder(hyper_p, run_options, NN, num_training_data, num_testing_data, data_train, labels_train, data_test, labels_test):
    #=== Loss functional ===#
    with tf.variable_scope('loss') as scope:
        auto_encoder_loss = tf.pow(tf.norm(NN.parameter_input_tf - NN.autoencoder_pred, 2, name= 'auto_encoder_loss'), 2)
        fwd_loss = hyper_p.penalty*tf.pow(tf.norm(NN.state_obs_tf - NN.forward_obs_pred, 2, name= 'fwd_loss'), 2)
        loss = tf.add(auto_encoder_loss, fwd_loss, name="loss")
        tf.summary.scalar("auto_encoder_loss",auto_encoder_loss)
        tf.summary.scalar("fwd_loss",fwd_loss)
        tf.summary.scalar("loss",loss)
        
    #=== Relative Error ===#
    with tf.variable_scope('relative_error') as scope:
        parameter_autoencoder_relative_error = tf.norm(NN.parameter_input_tf - NN.autoencoder_pred, 2)/tf.norm(NN.parameter_input_tf, 2)
        parameter_inverse_problem_relative_error = tf.norm(NN.parameter_input_tf - NN.inverse_pred, 2)/tf.norm(NN.parameter_input_tf, 2)
        state_obs_relative_error = tf.norm(NN.state_obs_tf - NN.forward_obs_pred, 2)/tf.norm(NN.state_obs_tf, 2)
        tf.summary.scalar("parameter_autoencoder_relative_error", parameter_autoencoder_relative_error)
        tf.summary.scalar("parameter_inverse_problem_relative_error", parameter_inverse_problem_relative_error)
        tf.summary.scalar("state_obs_relative_error", state_obs_relative_error)        
                
    #=== Set optimizers ===#
    with tf.variable_scope('Training') as scope:
        optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer_LBFGS = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                                 method='L-BFGS-B',
                                                                 options={'maxiter':10000,
                                                                          'maxfun':50000,
                                                                          'maxcor':50,
                                                                          'maxls':50,
                                                                          'ftol':1.0 * np.finfo(float).eps})
        #=== Track gradients ===#
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        gradients_tf = optimizer_Adam.compute_gradients(loss = loss)
        for gradient, variable in gradients_tf:
            tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient))
        optimizer_Adam_op = optimizer_Adam.apply_gradients(gradients_tf)
                    
    #=== Set GPU configuration options ===#
    gpu_options = tf.GPUOptions(visible_device_list=hyper_p.gpu,
                                allow_growth=True)
    
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=2,
                                gpu_options= gpu_options)
    
    #=== Tensorboard ===# type "tensorboard --logdir=Tensorboard" into terminal and click the link
    summ = tf.summary.merge_all()
    if os.path.exists('../Tensorboard/' + run_options.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + run_options.filename)  
    writer = tf.summary.FileWriter('../Tensorboard/' + run_options.filename)
    
    ########################
    #   Train Autoencoder  #
    ########################          
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.initialize_all_variables()) 
        writer.add_graph(sess.graph)
               
        #=== Train neural network ===#
        print('Beginning Training\n')
        num_batches = int(hyper_p.num_training_data/hyper_p.batch_size)
        for epoch in range(hyper_p.num_epochs):
            print('================================')
            print('            Epoch %d            ' %(epoch))
            print('================================')
            print(run_options.filename)
            print('GPU: ' + hyper_p.gpu + '\n')
            print('Optimizing %d batches of size %d:' %(num_batches, hyper_p.batch_size))
            start_time_epoch = time.time()
            minibatches = random_mini_batches(parameter_train.T, state_obs_train.T, hyper_p.batch_size, 1234)
            for batch_num in range(num_batches):
                parameter_train_batch = minibatches[batch_num][0].T
                state_obs_train_batch = minibatches[batch_num][1].T
                start_time_batch = time.time()
                sess.run(optimizer_Adam_op, feed_dict = {NN.parameter_input_tf: parameter_train_batch, NN.state_obs_tf: state_obs_train_batch})
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
                
            #=== Display Batch Iteration Information ===#
            elapsed_time_epoch = time.time() - start_time_epoch
            loss_value = sess.run(loss, feed_dict = {NN.parameter_input_tf: parameter_train_batch, NN.state_obs_tf: state_obs_train_batch}) 
            autoencoder_RE, parameter_RE, state_RE, s = sess.run([parameter_autoencoder_relative_error, parameter_inverse_problem_relative_error, state_obs_relative_error, summ], \
                                                                 feed_dict = {NN.parameter_input_tf: parameter_test, NN.state_obs_tf: state_obs_test, NN.state_obs_inverse_input_tf: state_obs_test})
            #accuracy = compute_test_accuracy(sess, NN, test_accuracy, num_testing_data, hyper_p.batch_size, data_test, labels_test)
            #s = sess.run(summ, feed_dict = {NN.data_tf: data_test, NN.labels_tf: labels_test}) 
            writer.add_summary(s, epoch)
            print('Time per Epoch: %.2f' %(elapsed_time_epoch))
            print('Loss: %.3e, Relative Errors: Autoencoder: %.3e, Parameter: %.3e, State: %.3e\n' %(loss_value, autoencoder_RE, parameter_RE, state_RE))
            start_time_epoch = time.time() 
                                             
            #=== Optimize with LBFGS ===#
            if run_options.use_LBFGS == 1:
                print('Optimizing with LBFGS')  
                start_time_LBFGS = time.time()
                optimizer_LBFGS.minimize(sess, feed_dict = {NN.parameter_input_tf: parameter_train_batch, NN.state_obs_tf: state_obs_train_batch})
                time_elapsed_LBFGS = time.time() - start_time_LBFGS 
                loss_value = sess.run(loss, feed_dict = {NN.parameter_input_tf: parameter_train_batch, NN.state_obs_tf: state_obs_train_batch}) 
                autoencoder_RE, parameter_RE, state_RE, s = sess.run([parameter_autoencoder_relative_error, parameter_inverse_problem_relative_error, state_obs_relative_error, summ], \
                                                                     feed_dict = {NN.parameter_input_tf: parameter_test, NN.state_obs_tf: state_obs_test, NN.state_obs_inverse_input_tf: state_obs_test})
                writer.add_summary(s, epoch)
                print('LBFGS Optimization Complete')   
                print('Time for LBFGS: %.2f' %(time_elapsed_LBFGS))
                print('Loss: %.3e, Relative Errors: Autoencoder: %.3e, Parameter: %.3e, State: %.3e\n' %(loss_value, autoencoder_RE, parameter_RE, state_RE))
        
        #=== Save final model ===#
        save_weights_and_biases(sess, hyper_p.truncation_layer, NN.layers, run_options.NN_savefile_name)  
        print('Final Model Saved')  
