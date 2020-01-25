#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:53:31 2019

@author: hwan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:31:44 2019

@author: hwan
"""
import sys
sys.path.append('../..')

import shutil # for deleting directories
import os
import time

import tensorflow as tf
import numpy as np
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths, NN, obs_indices, loss_autoencoder, loss_encoder, loss_fenics, relative_error, data_and_latent_train, data_and_latent_val, data_and_latent_test, data_dimension, num_batches_train):
    #=== Generate Dolfin Function Space and Mesh ===#
    if run_options.fin_dimensions_2D == 1:
        V, mesh = get_space_2D(40)
    if run_options.fin_dimensions_3D == 1:    
        V, mesh = get_space_3D(40)
    solver = Fin(V)
    print(V.dim())  
    
    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()

    #=== Define Metrics ===#
    mean_loss_train = tf.keras.metrics.Mean()
    mean_loss_train_autoencoder = tf.keras.metrics.Mean() 
    mean_loss_train_encoder = tf.keras.metrics.Mean()
    mean_loss_train_fenics = tf.keras.metrics.Mean()
    
    mean_loss_val = tf.keras.metrics.Mean()
    mean_loss_val_autoencoder = tf.keras.metrics.Mean()
    mean_loss_val_encoder = tf.keras.metrics.Mean()
    mean_loss_val_fenics = tf.keras.metrics.Mean()
    
    mean_loss_test = tf.keras.metrics.Mean()
    mean_loss_test_autoencoder = tf.keras.metrics.Mean()
    mean_loss_test_encoder = tf.keras.metrics.Mean()
    mean_loss_test_fenics = tf.keras.metrics.Mean()    
    
    mean_relative_error_data_autoencoder = tf.keras.metrics.Mean()
    mean_relative_error_latent_encoder = tf.keras.metrics.Mean()
    mean_relative_error_data_decoder = tf.keras.metrics.Mean()
    
    #=== Initialize Metric Storage Arrays ===#
    storage_array_loss_train = np.array([])
    storage_array_loss_train_autoencoder = np.array([])
    storage_array_loss_train_encoder = np.array([])
    storage_array_loss_train_fenics = np.array([])
    
    storage_array_loss_val = np.array([])
    storage_array_loss_val_autoencoder = np.array([])
    storage_array_loss_val_encoder = np.array([])
    storage_array_loss_val_fenics = np.array([])
    
    storage_array_loss_test = np.array([])
    storage_array_loss_test_autoencoder = np.array([])
    storage_array_loss_test_encoder = np.array([])
    storage_array_loss_test_fenics = np.array([])
    
    storage_array_relative_error_data_autoencoder = np.array([])
    storage_array_relative_error_latent_encoder = np.array([])
    storage_array_relative_error_data_decoder = np.array([])
    
    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)
    
    #=== Tensorboard ===# Tensorboard: type "tensorboard --logdir=Tensorboard" into terminal and click the link
    if os.path.exists(file_paths.tensorboard_directory): # Remove existing directory because Tensorboard graphs mess up of you write over it
        shutil.rmtree(file_paths.tensorboard_directory)  
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    #=== Train Step ===#
    #@tf.function
    def train_step(batch_data_train, batch_latent_train):
        with tf.GradientTape() as tape:
            batch_data_pred_train_AE = NN(batch_data_train)
            batch_latent_pred_train = NN.encoder(batch_data_train)
            batch_loss_train_autoencoder = loss_autoencoder(batch_data_pred_train_AE, batch_data_train)                    
            batch_loss_train_encoder = loss_encoder(batch_latent_pred_train, batch_latent_train, hyperp.penalty)
            if file_paths.autoencoder_type == 'rev_':
                batch_state_obs_train = batch_data_train
                batch_parameter_pred = NN.encoder(batch_data_train)
            else:
                batch_state_obs_train = batch_latent_train
                batch_parameter_pred = batch_data_pred_train_AE
            batch_loss_train_fenics = loss_fenics(hyperp, run_options, V, solver, obs_indices, batch_state_obs_train, batch_parameter_pred, hyperp.penalty_aug)
            batch_loss_train_NN = batch_loss_train_autoencoder + batch_loss_train_encoder
            batch_loss_train = batch_loss_train_autoencoder + batch_loss_train_encoder + batch_loss_train_fenics
        gradients_fenics = fenics_gradient(hyperp, run_options, V, solver, obs_indices, batch_state_obs_train, batch_parameter_pred, hyperp.penalty_aug)
        gradients_NN = tape.gradient(batch_loss_train_NN, NN.trainable_variables)
        pdb.set_trace()
        optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
        mean_loss_train(batch_loss_train)
        mean_loss_train_autoencoder(batch_loss_train_autoencoder)
        mean_loss_train_encoder(batch_loss_train_encoder)
        mean_loss_train_fenics(batch_loss_train_fenics)
        return gradients

    #=== Validation Step ===#
    #@tf.function
    def val_step(batch_data_val, batch_latent_val):
        batch_data_pred_val_AE = NN(batch_data_val)
        batch_latent_pred_val = NN.encoder(batch_data_val)
        batch_loss_val_autoencoder = loss_autoencoder(batch_data_pred_val_AE, batch_data_val)
        batch_loss_val_encoder = loss_encoder(batch_latent_pred_val, batch_latent_val, hyperp.penalty)
        if file_paths.autoencoder_type == 'rev_':
            batch_state_obs_val = batch_data_val
            batch_parameter_pred = NN.encoder(batch_data_val)
        else:
            batch_state_obs_val = batch_latent_val
            batch_parameter_pred = batch_data_pred_val_AE
        batch_loss_val_fenics = loss_fenics(hyperp, run_options, V, solver, obs_indices, batch_state_obs_val, batch_parameter_pred, hyperp.penalty_aug)
        batch_loss_val = batch_loss_val_autoencoder + batch_loss_val_encoder + batch_loss_val_fenics
        mean_loss_val_autoencoder(batch_loss_val_autoencoder)
        mean_loss_val_encoder(batch_loss_val_encoder)
        mean_loss_val_fenics(batch_loss_val_fenics)
        mean_loss_val(batch_loss_val)     
    
    #=== Test Step ===#
    #@tf.function
    def test_step(batch_data_test, batch_latent_test):
        batch_data_pred_test_AE = NN(batch_data_test)
        batch_data_pred_test_decoder = NN.decoder(batch_latent_test)
        batch_latent_pred_test = NN.encoder(batch_data_test)
        batch_loss_test_autoencoder = loss_autoencoder(batch_data_pred_test_AE, batch_data_test)
        batch_loss_test_encoder = loss_encoder(batch_latent_pred_test, batch_latent_test, hyperp.penalty)
        if file_paths.autoencoder_type == 'rev_':
            batch_state_obs_test = batch_data_test
            batch_parameter_pred = NN.encoder(batch_data_test)
        else:
            batch_state_obs_test = batch_latent_test
            batch_parameter_pred = batch_data_pred_test_AE
        batch_loss_test_fenics = loss_fenics(hyperp, run_options, V, solver, obs_indices, batch_state_obs_test, batch_parameter_pred, hyperp.penalty_aug)
        batch_loss_test = batch_loss_test_autoencoder + batch_loss_test_encoder + batch_loss_test_fenics
        mean_loss_test_autoencoder(batch_loss_test_autoencoder)
        mean_loss_test_encoder(batch_loss_test_encoder)
        mean_loss_test_fenics(batch_loss_test_fenics)
        mean_loss_test(batch_loss_test)
        mean_relative_error_data_autoencoder(relative_error(batch_data_pred_test_AE, batch_data_test))
        mean_relative_error_latent_encoder(relative_error(batch_latent_pred_test, batch_latent_test))
        mean_relative_error_data_decoder(relative_error(batch_data_pred_test_decoder, batch_data_test))
        
###############################################################################
#                             Train Neural Network                            #
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
        for batch_num, (batch_data_train, batch_latent_train) in data_and_latent_train.enumerate():
            start_time_batch = time.time()
            gradients = train_step(batch_data_train, batch_latent_train)
            elapsed_time_batch = time.time() - start_time_batch
            #=== Display Model Summary ===#
            if batch_num == 0 and epoch == 0:
                NN.summary()
            if batch_num  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))
        
        #=== Computing Relative Errors Validation ===#
        for batch_data_val, batch_latent_val in data_and_latent_val:
            val_step(batch_data_val, batch_latent_val)
            
        #=== Computing Relative Errors Test ===#
        for batch_data_test, batch_latent_test in data_and_latent_test:
            test_step(batch_data_test, batch_latent_test)

        #=== Track Training Metrics, Weights and Gradients ===#
        with summary_writer.as_default():
            tf.summary.scalar('loss_training', mean_loss_train.result(), step=epoch)
            tf.summary.scalar('loss_training_autoencoder', mean_loss_train_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_training_encoder', mean_loss_train_encoder.result(), step=epoch)
            tf.summary.scalar('loss_training_fenics', mean_loss_train_fenics.result(), step=epoch)
            tf.summary.scalar('loss_val', mean_loss_val.result(), step=epoch)
            tf.summary.scalar('loss_val_autoencoder', mean_loss_val_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_val_encoder', mean_loss_val_encoder.result(), step=epoch)
            tf.summary.scalar('loss_val_fenics', mean_loss_val_fenics.result(), step=epoch)
            tf.summary.scalar('loss_test', mean_loss_test.result(), step=epoch)
            tf.summary.scalar('loss_test_autoencoder', mean_loss_test_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_test_encoder', mean_loss_test_encoder.result(), step=epoch)
            tf.summary.scalar('loss_test_fenics', mean_loss_test_fenics.result(), step=epoch)
            tf.summary.scalar('relative_error_data_autoencoder', mean_relative_error_data_autoencoder.result(), step=epoch)
            tf.summary.scalar('relative_error_data_decoder', mean_relative_error_data_decoder.result(), step=epoch)
            tf.summary.scalar('relative_error_latent_encoder', mean_relative_error_latent_encoder.result(), step=epoch)
            for w in NN.weights:
                tf.summary.histogram(w.name, w, step=epoch)
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in zip(gradients, NN.trainable_variables):
                tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)              
                
        #=== Update Storage Arrays ===#
        storage_array_loss_train = np.append(storage_array_loss_train, mean_loss_train.result())
        storage_array_loss_train_autoencoder = np.append(storage_array_loss_train_autoencoder, mean_loss_train_autoencoder.result())
        storage_array_loss_train_encoder = np.append(storage_array_loss_train_encoder, mean_loss_train_encoder.result())
        storage_array_loss_train_fenics = np.append(storage_array_loss_train_fenics, mean_loss_train_fenics.result())
        storage_array_loss_val = np.append(storage_array_loss_val, mean_loss_val.result())
        storage_array_loss_val_autoencoder = np.append(storage_array_loss_val_autoencoder, mean_loss_val_autoencoder.result())
        storage_array_loss_val_encoder = np.append(storage_array_loss_val_encoder, mean_loss_val_encoder.result())
        storage_array_loss_val_fenics = np.append(storage_array_loss_val_fenics, mean_loss_val_fenics.result())
        storage_array_loss_test = np.append(storage_array_loss_test, mean_loss_test.result())
        storage_array_loss_test_autoencoder = np.append(storage_array_loss_test_autoencoder, mean_loss_test_autoencoder.result())
        storage_array_loss_test_encoder = np.append(storage_array_loss_test_encoder, mean_loss_test_encoder.result())
        storage_array_loss_test_fenics = np.append(storage_array_loss_test_fenics, mean_loss_test_fenics.result())
        storage_array_relative_error_data_autoencoder = np.append(storage_array_relative_error_data_autoencoder, mean_relative_error_data_autoencoder.result())
        storage_array_relative_error_latent_encoder = np.append(storage_array_relative_error_latent_encoder, mean_relative_error_latent_encoder.result())
        storage_array_relative_error_data_decoder = np.append(storage_array_relative_error_data_decoder, mean_relative_error_data_decoder.result())
            
        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Aug: %.3e' %(mean_loss_train.result(), mean_loss_train_autoencoder.result(), mean_loss_train_encoder.result(), mean_loss_train_fenics.result()))
        print('Val Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Aug: %.3e' %(mean_loss_val.result(), mean_loss_val_autoencoder.result(), mean_loss_val_encoder.result(), mean_loss_val_fenics.result()))
        print('Test Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Aug: %.3e' %(mean_loss_test.result(), mean_loss_test_autoencoder.result(), mean_loss_test_encoder.result(), mean_loss_test_fenics.result()))
        print('Rel Errors: AE: %.3e, Encoder: %.3e, Decoder: %.3e\n' %(mean_relative_error_data_autoencoder.result(), mean_relative_error_latent_encoder.result(), mean_relative_error_data_decoder.result()))
        start_time_epoch = time.time()
        
        #=== Resetting Metrics ===#
        mean_loss_train.reset_states()
        mean_loss_train_autoencoder.reset_states()
        mean_loss_train_encoder.reset_states()
        mean_loss_train_fenics.reset_states()
        mean_loss_val.reset_states()
        mean_loss_val_autoencoder.reset_states()
        mean_loss_val_encoder.reset_states()    
        mean_loss_val_fenics.reset_states()    
        mean_loss_test.reset_states()
        mean_loss_test_autoencoder.reset_states()
        mean_loss_test_encoder.reset_states()
        mean_loss_test_fenics.reset_states()
        mean_relative_error_data_autoencoder.reset_states()
        mean_relative_error_latent_encoder.reset_states()
        mean_relative_error_data_decoder.reset_states()
            
    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 
    
    return storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_encoder, storage_array_loss_train_fenics, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_encoder, storage_array_loss_val_fenics, storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_encoder, storage_array_loss_test_fenics, storage_array_relative_error_data_autoencoder, storage_array_relative_error_latent_encoder, storage_array_relative_error_data_decoder 