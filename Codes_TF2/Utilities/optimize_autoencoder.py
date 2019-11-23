#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:31:44 2019

@author: hwan
"""
import shutil # for deleting directories
import os
import time

import tensorflow as tf
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths, NN, loss_autoencoder, loss_forward_problem, relative_error, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, parameter_dimension, num_batches_train):
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()

    #=== Define Metrics ===#
    loss_train_batch_average = tf.keras.metrics.Mean()
    loss_train_batch_average_autoencoder = tf.keras.metrics.Mean() 
    loss_train_batch_average_forward_problem = tf.keras.metrics.Mean()
    
    loss_val_batch_average = tf.keras.metrics.Mean()
    loss_val_batch_average_autoencoder = tf.keras.metrics.Mean()
    loss_val_batch_average_forward_problem = tf.keras.metrics.Mean()
    
    loss_test_batch_average = tf.keras.metrics.Mean()
    loss_test_batch_average_autoencoder = tf.keras.metrics.Mean()
    loss_test_batch_average_forward_problem = tf.keras.metrics.Mean()    
    
    relative_error_batch_average_parameter_autoencoder = tf.keras.metrics.Mean()
    relative_error_batch_average_parameter_inverse_problem = tf.keras.metrics.Mean()
    relative_error_batch_average_state_obs = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
    storage_array_loss_train = np.array([])
    storage_array_loss_train_autoencoder = np.array([])
    storage_array_loss_train_forward_problem = np.array([])
    
    storage_array_loss_val = np.array([])
    storage_array_loss_val_autoencoder = np.array([])
    storage_array_loss_val_forward_problem = np.array([])
    
    storage_array_loss_test = np.array([])
    storage_array_loss_test_autoencoder = np.array([])
    storage_array_loss_test_forward_problem = np.array([])
    
    storage_array_relative_error_parameter_autoencoder = np.array([])
    storage_array_relative_error_parameter_inverse_problem = np.array([])
    storage_array_relative_error_state_obs = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists(file_paths.tensorboard_directory): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree(file_paths.tensorboard_directory)  
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

###############################################################################
#                                Training Step                                #
###############################################################################
    @tf.function
    def train_step(parameter_train, state_obs_train, loss_autoencoder, loss_forward_problem):
        with tf.GradientTape() as tape:
            parameter_pred_train_AE = NN(parameter_train)
            state_pred_train = NN.encoder(parameter_train)
            loss_train_batch_autoencoder = loss_autoencoder(parameter_pred_train_AE, parameter_train)
            loss_train_batch_forward_problem = loss_forward_problem(state_pred_train, state_obs_train, hyperp.penalty)
            loss_train_batch = loss_train_batch_autoencoder + loss_train_batch_forward_problem
            gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            return loss_train_batch, loss_train_batch_autoencoder, loss_train_batch_forward_problem, gradients

###############################################################################
#                          Train Neural Network                               #
############################################################################### 
    print('Beginning Training')
    for epoch in range(hyperp.num_epochs):
        print('================================')
        print('            Epoch %d            ' %(epoch))
        print('================================')
        print(file_paths.filename)
        print('GPU: ' + run_options.which_gpu + '\n')
        print('Optimizing %d batches of size %d:' %(num_batches_train, hyperp.batch_size))
        start_time_epoch = time.time()
        for batch_num, (parameter_train, state_obs_train) in parameter_and_state_obs_train.enumerate():
            start_time_batch = time.time()
            loss_train_batch, loss_train_batch_autoencoder, loss_train_batch_forward_problem, gradients\
            = train_step(parameter_train, state_obs_train, loss_autoencoder, loss_forward_problem)
            elapsed_time_batch = time.time() - start_time_batch
            #=== Display Model Summary ===#
            if batch_num == 0 and epoch == 0:
                NN.summary()
            if batch_num  == 0:
                print('Time per Batch: %.2f' %(elapsed_time_batch))
# =============================================================================
#             with tf.GradientTape() as tape:
#                 parameter_pred_train_AE = NN(parameter_train)
#                 state_pred_train = NN.encoder(parameter_train)
#                 #=== Display Model Summary ===#
#                 if batch_num == 0 and epoch == 0:
#                     NN.summary()
#                 loss_train_batch_autoencoder = loss_autoencoder(parameter_pred_train_AE, parameter_train)
#                 loss_train_batch_forward_problem = loss_forward_problem(state_pred_train, state_obs_train, hyperp.penalty)
#                 loss_train_batch = loss_train_batch_autoencoder + loss_train_batch_forward_problem
#                 gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
#                 elapsed_time_batch = time.time() - start_time_batch
#                 if batch_num  == 0:
#                     print('Time per Batch: %.2f' %(elapsed_time_batch))
#             loss_train_batch_average(loss_train_batch) 
# =============================================================================
            loss_train_batch_average_autoencoder(loss_train_batch_autoencoder)
            loss_train_batch_average_forward_problem(loss_train_batch_forward_problem)
        
        #=== Computing Relative Errors Validation ===#
        for parameter_val, state_obs_val in parameter_and_state_obs_val:
            parameter_pred_val_batch_AE = NN(parameter_val)
            state_pred_val_batch = NN.encoder(parameter_val)
            loss_val_batch_autoencoder = loss_autoencoder(parameter_pred_val_batch_AE, parameter_val)
            loss_val_batch_forward_problem = loss_forward_problem(state_pred_val_batch, state_obs_val, hyperp.penalty)
            loss_val_batch = loss_val_batch_autoencoder + loss_val_batch_forward_problem
            loss_val_batch_average(loss_val_batch)
            loss_val_batch_average_autoencoder(loss_val_batch_autoencoder)
            loss_val_batch_average_forward_problem(loss_val_batch_forward_problem)
            
        #=== Computing Relative Errors Test ===#
        for parameter_test, state_obs_test in parameter_and_state_obs_test:
            parameter_pred_test_batch_AE = NN(parameter_test)
            parameter_pred_test_batch_Inverse_problem = NN.decoder(state_obs_test)
            state_pred_test_batch = NN.encoder(parameter_test)
            loss_test_batch_autoencoder = loss_autoencoder(parameter_pred_test_batch_AE, parameter_test)
            loss_test_batch_forward_problem = loss_forward_problem(state_pred_test_batch, state_obs_test, hyperp.penalty)
            loss_test_batch = loss_test_batch_autoencoder + loss_test_batch_forward_problem
            loss_test_batch_average(loss_test_batch)
            loss_test_batch_average_autoencoder(loss_test_batch_autoencoder)
            loss_test_batch_average_forward_problem(loss_test_batch_forward_problem)
            relative_error_batch_parameter_autoencoder = relative_error(parameter_pred_test_batch_AE, parameter_test)
            relative_error_batch_parameter_inverse_problem = relative_error(parameter_pred_test_batch_Inverse_problem, parameter_test)
            relative_error_batch_state_obs = relative_error(state_pred_test_batch, state_obs_test)
            relative_error_batch_average_parameter_autoencoder(relative_error_batch_parameter_autoencoder)
            relative_error_batch_average_parameter_inverse_problem(relative_error_batch_parameter_inverse_problem)
            relative_error_batch_average_state_obs(relative_error_batch_state_obs)

        #=== Track Training Metrics, Weights and Gradients ===#
        with summary_writer.as_default():
            tf.summary.scalar('loss_training', loss_train_batch_average.result(), step=epoch)
            tf.summary.scalar('loss_training_autoencoder', loss_train_batch_average_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_training_forward_problem', loss_train_batch_average_forward_problem.result(), step=epoch)
            tf.summary.scalar('loss_val', loss_val_batch_average.result(), step=epoch)
            tf.summary.scalar('loss_val_autoencoder', loss_val_batch_average_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_val_forward_problem', loss_val_batch_average_forward_problem.result(), step=epoch)
            tf.summary.scalar('loss_test', loss_test_batch_average.result(), step=epoch)
            tf.summary.scalar('loss_test_autoencoder', loss_test_batch_average_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_test_forward_problem', loss_test_batch_average_forward_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_autoencoder', relative_error_batch_average_parameter_autoencoder.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_inverse_problem', relative_error_batch_average_parameter_inverse_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_state_obs', relative_error_batch_average_state_obs.result(), step=epoch)
            storage_array_loss_train = np.append(storage_array_loss_train, loss_train_batch_average.result())
            storage_array_loss_train_autoencoder = np.append(storage_array_loss_train_autoencoder, loss_train_batch_average_autoencoder.result())
            storage_array_loss_train_forward_problem = np.append(storage_array_loss_train_forward_problem, loss_train_batch_average_forward_problem.result())
            storage_array_loss_val = np.append(storage_array_loss_val, loss_val_batch_average.result())
            storage_array_loss_val_autoencoder = np.append(storage_array_loss_val_autoencoder, loss_val_batch_average_autoencoder.result())
            storage_array_loss_val_forward_problem = np.append(storage_array_loss_val_forward_problem, loss_val_batch_average_forward_problem.result())
            storage_array_loss_test = np.append(storage_array_loss_test, loss_test_batch_average.result())
            storage_array_loss_test_autoencoder = np.append(storage_array_loss_test_autoencoder, loss_test_batch_average_autoencoder.result())
            storage_array_loss_test_forward_problem = np.append(storage_array_loss_test_forward_problem, loss_test_batch_average_forward_problem.result())
            storage_array_relative_error_parameter_autoencoder = np.append(storage_array_relative_error_parameter_autoencoder, relative_error_batch_average_parameter_autoencoder.result())
            storage_array_relative_error_parameter_inverse_problem = np.append(storage_array_relative_error_parameter_inverse_problem, relative_error_batch_average_parameter_inverse_problem.result())
            storage_array_relative_error_state_obs = np.append(storage_array_relative_error_state_obs, relative_error_batch_average_state_obs.result())
            for w in NN.weights:
                tf.summary.histogram(w.name, w, step=epoch)
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in zip(gradients, NN.trainable_variables):
                tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)
            
        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_train_batch_average.result(), loss_train_batch_average_autoencoder.result(), loss_train_batch_average_forward_problem.result()))
        print('Val Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_val_batch_average.result(), loss_val_batch_average_autoencoder.result(), loss_val_batch_average_forward_problem.result()))
        print('Test Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_test_batch_average.result(), loss_test_batch_average_autoencoder.result(), loss_test_batch_average_forward_problem.result()))
        print('Rel Errors: AE: %.3e, Inverse: %.3e, Forward: %.3e\n' %(relative_error_batch_average_parameter_autoencoder.result(), relative_error_batch_average_parameter_inverse_problem.result(), relative_error_batch_average_state_obs.result()))
        start_time_epoch = time.time()   
            
    #=== Save final model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 

    return storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs 