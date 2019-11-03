#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:31:44 2019

@author: hwan
"""
import tensorflow as tf
import numpy as np

import shutil # for deleting directories
import os
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                          Loss and Relative Errors                           #
###############################################################################
def loss_autoencoder(autoencoder_pred, parameter_true):
    return tf.reduce_mean(tf.square(tf.subtract(parameter_true, autoencoder_pred)))

def loss_forward_problem(state_obs_pred, state_obs_true, penalty):
    return penalty*tf.reduce_mean(tf.square(state_obs_pred, state_obs_true))

def relative_error(prediction, true):
    return tf.norm(true - prediction, 2)/tf.norm(true, 2)

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyper_p, run_options, NN, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, parameter_dimension, num_batches_train):
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()

    #=== Define Metrics ===#
    loss_train_batch_average = tf.keras.metrics.Mean()
    loss_train_batch_average_autoencoder = tf.keras.metrics.Mean() 
    loss_train_batch_average_forward_problem = tf.keras.metrics.Mean()
    
    loss_val_batch_average = tf.keras.metrics.Mean()
    loss_val_batch_average_autoencoder = tf.keras.metrics.Mean()
    loss_val_batch_average_forward_problem = tf.keras.metrics.Mean()
    
    relative_error_parameter_autoencoder = tf.keras.metrics.Mean()
    relative_error_parameter_inverse_problem = tf.keras.metrics.Mean()
    relative_error_state_obs = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
    storage_array_loss_train = np.array([])
    storage_array_loss_train_autoencoder = np.array([])
    storage_array_loss_train_forward_problem = np.array([])
    
    storage_array_loss_val = np.array([])
    storage_array_loss_val_autoencoder = np.array([])
    storage_array_loss_val_forward_problem = np.array([])
    
    storage_array_relative_error_parameter_autoencoder = np.array([])
    storage_array_relative_error_parameter_inverse_problem = np.array([])
    storage_array_relative_error_state_obs = np.array([])
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists('../Tensorboard/' + run_options.filename): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree('../Tensorboard/' + run_options.filename)  
    summary_writer = tf.summary.create_file_writer('../Tensorboard/' + run_options.filename)

###############################################################################
#                          Train Neural Network                               #
############################################################################### 
    print('Beginning Training')
    for epoch in range(hyper_p.num_epochs):
        print('================================')
        print('            Epoch %d            ' %(epoch))
        print('================================')
        print(run_options.filename)
        print('GPU: ' + hyper_p.gpu + '\n')
        print('Optimizing %d batches of size %d:' %(num_batches_train, hyper_p.batch_size))
        start_time_epoch = time.time()
        for batch_num, (parameter_train, state_obs_train) in parameter_and_state_obs_train.enumerate():
            with tf.GradientTape() as tape:
                start_time_batch = time.time()
                parameter_pred_train_AE = NN(parameter_train)
                #=== Display Model Summary ===#
                if batch_num == 0 and epoch == 0:
                    NN.summary()
                loss_train_batch_autoencoder = loss_autoencoder(parameter_pred_train_AE, parameter_train)
                loss_train_batch_forward_problem = loss_forward_problem( , state_obs_train)
                loss_train_batch = loss_train_batch_autoencoder + loss_train_batch_forward_problem
                gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
                optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
                elapsed_time_batch = time.time() - start_time_batch
                if batch_num  == 0:
                    print('Time per Batch: %.2f' %(elapsed_time_batch))
            loss_train_batch_average(loss_train_batch) 
            loss_train_batch_average_autoencoder(loss_train_batch_autoencoder)
            loss_train_batch_average_forward_problem(loss_train_batch_forward_problem)
        
        #=== Computing Relative Errors ===#
        for parameter_val, state_obs_val in parameter_and_state_obs_val:
            parameter_pred_val_AE = NN(parameter_val)
            loss_val_batch_autoencoder = loss_autoencoder(parameter_pred_val_AE, parameter_val)
            loss_val_batch_forward_problem = loss_forward_problem( , state_obs_val)
            loss_val_batch = loss_val_batch_autoencoder + loss_val_batch_forward_problem
            loss_val_batch_average(loss_val_batch)
            loss_val_batch_average_autoencoder(loss_val_batch_autoencoder)
            loss_val_batch_average_forward_problem(loss_val_batch_forward_problem)
            relative_error_parameter_autoencoder = relative_error(parameter_pred_val_AE, parameter_val)
            relative_error_parameter_inverse_problem = relative_error( , parameter_val)
            relative_error_state_obs = relative_error( , state_obs_val)
        
        #=== Track Training Metrics, Weights and Gradients ===#
        with summary_writer.as_default():
            tf.summary.scalar('loss_training', loss_train_batch_average.result(), step=epoch)
            tf.summary.scalar('loss_training_autoencoder', loss_train_batch_average_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_training_forward_problem', loss_train_batch_average_forward_problem.result(), step=epoch)
            tf.summary.scalar('loss_val', loss_val_batch_average.result(), step=epoch)
            tf.summary.scalar('loss_val_autoencoder', loss_val_batch_average_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_val_forward_problem', loss_val_batch_average_forward_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_autoencoder', relative_error_parameter_autoencoder.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_inverse_problem', relative_error_parameter_inverse_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_state_obs', relative_error_state_obs.result(), step=epoch)
            storage_array_loss_train = np.append(storage_array_loss_train, loss_train_batch_average.result())
            storage_array_loss_train_autoencoder = np.append(storage_array_loss_train_autoencoder, loss_train_batch_average_autoencoder.result())
            storage_array_loss_train_forward_problem = np.append([])
            storage_array_loss_val = np.append(storage_array_loss_val, loss_val_batch_average.result())
            storage_array_loss_val_autoencoder = np.append(storage_array_loss_val_autoencoder, loss_val_batch_average_autoencoder.result())
            storage_array_loss_val_forward_problem = np.append(storage_array_loss_val_forward_problem, loss_val_batch_average_forward_problem.result())
            storage_array_relative_error_parameter_autoencoder = np.append(storage_array_relative_error_parameter_autoencoder, relative_error_parameter_autoencoder.result())
            storage_array_relative_error_parameter_inverse_problem = np.append(storage_array_relative_error_parameter_inverse_problem, relative_error_parameter_inverse_problem.result())
            storage_array_relative_error_state_obs = np.append(storage_array_relative_error_state_obs, relative_error_state_obs.result())
            for w in NN.weights:
                tf.summary.histogram(w.name, w, step=epoch)
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in zip(gradients, NN.trainable_variables):
                tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)
            
        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.2f\n' %(elapsed_time_epoch))
        print('Training Set: Loss: %.3e, Accuracy: %.3f' %(loss_train_batch_average.result(), accuracy_train_batch_average.result()))
        print('Validation Set: Loss: %.3e, Accuracy: %.3f\n' %(loss_val_batch_average.result(), accuracy_val_batch_average.result()))
        start_time_epoch = time.time()   
            
    #=== Save final model ===#
    NN.save_weights(run_options.NN_savefile_name)
    print('Final Model Saved') 
    
    return storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs 