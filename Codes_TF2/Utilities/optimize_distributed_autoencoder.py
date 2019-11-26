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
import shutil # for deleting directories
import os
import time

import tensorflow as tf
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE, hyperp, run_options, file_paths, NN, loss_autoencoder, loss_forward_problem, relative_error, parameter_and_state_obs_train, parameter_and_state_obs_val, parameter_and_state_obs_test, parameter_dimension, num_batches_train):
    #=== Check Number of Parallel Computations and Set Global Batch Size ===#
    print('Number of Replicas in Sync: %d' %(dist_strategy.num_replicas_in_sync))   
    
    #=== Distribute Data ===#
    dist_parameter_and_state_obs_train = dist_strategy.experimental_distribute_dataset(parameter_and_state_obs_train)
    dist_parameter_and_state_obs_val = dist_strategy.experimental_distribute_dataset(parameter_and_state_obs_val)
    dist_parameter_and_state_obs_test = dist_strategy.experimental_distribute_dataset(parameter_and_state_obs_test)

    with dist_strategy.scope():    
        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Define Metrics ===#        
        mean_loss_train_autoencoder = tf.keras.metrics.Mean() 
        mean_loss_train_forward_problem = tf.keras.metrics.Mean()
        
        mean_loss_val = tf.keras.metrics.Mean()
        mean_loss_val_autoencoder = tf.keras.metrics.Mean()
        mean_loss_val_forward_problem = tf.keras.metrics.Mean()
        
        mean_loss_test = tf.keras.metrics.Mean()
        mean_loss_test_autoencoder = tf.keras.metrics.Mean()
        mean_loss_test_forward_problem = tf.keras.metrics.Mean()    
        
        mean_relative_error_parameter_autoencoder = tf.keras.metrics.Mean()
        mean_relative_error_parameter_inverse_problem = tf.keras.metrics.Mean()
        mean_relative_error_state_obs = tf.keras.metrics.Mean()
    
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
#                   Training, Validation and Testing Step                     #
###############################################################################
    with dist_strategy.scope():
        #=== Training Step ===#
        def train_step(batch_parameter_train, batch_state_obs_train):
            with tf.GradientTape() as tape:
                batch_parameter_pred_train_AE = NN(batch_parameter_train)
                batch_state_pred_train = NN.encoder(batch_parameter_train)
                unscaled_replica_batch_loss_train_autoencoder = loss_autoencoder(batch_parameter_pred_train_AE, batch_parameter_train)
                unscaled_replica_batch_loss_train_forward_problem = loss_forward_problem(batch_state_pred_train, batch_state_obs_train, hyperp.penalty)
                unscaled_replica_batch_loss_train_batch = unscaled_replica_batch_loss_train_autoencoder + unscaled_replica_batch_loss_train_forward_problem
                scaled_replica_batch_loss_train = tf.reduce_sum(unscaled_replica_batch_loss_train_batch * (1./GLOBAL_BATCH_SIZE))
            gradients = tape.gradient(scaled_replica_batch_loss_train, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            mean_loss_train_autoencoder(unscaled_replica_batch_loss_train_autoencoder)
            mean_loss_train_forward_problem(unscaled_replica_batch_loss_train_batch)
            return scaled_replica_batch_loss_train
        
        @tf.function
        def dist_train_step(batch_parameter_train, batch_state_obs_train):
            per_replica_losses = dist_strategy.experimental_run_v2(train_step, args=(batch_parameter_train, batch_state_obs_train))
            return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                        
        #=== Validation Step ===#
        def val_step(batch_parameter_val, batch_state_obs_val):
            batch_parameter_pred_val_AE = NN(batch_parameter_val)
            batch_state_pred_val = NN.encoder(batch_parameter_val)
            unscaled_replica_batch_loss_val_autoencoder = loss_autoencoder(batch_parameter_pred_val_AE, batch_parameter_val)
            unscaled_replica_batch_loss_val_forward_problem = loss_forward_problem(batch_state_pred_val, batch_state_obs_val, hyperp.penalty)
            unscaled_replica_batch_loss_val = unscaled_replica_batch_loss_val_autoencoder + unscaled_replica_batch_loss_val_forward_problem
            mean_loss_val_autoencoder(unscaled_replica_batch_loss_val_autoencoder)
            mean_loss_val_forward_problem(unscaled_replica_batch_loss_val_forward_problem)
            mean_loss_val(unscaled_replica_batch_loss_val)
        
        @tf.function
        def dist_val_step(batch_parameter_val, batch_state_obs_val):
            return dist_strategy.experimental_run_v2(val_step, (batch_parameter_val, batch_state_obs_val))
        
        #=== Test Step ===#
        def test_step(batch_parameter_test, batch_state_obs_test):
            batch_parameter_pred_test_AE = NN(batch_parameter_test)
            batch_parameter_pred_test_Inverse_problem = NN.decoder(batch_state_obs_test)
            batch_state_pred_test = NN.encoder(batch_parameter_test)
            unscaled_replica_batch_loss_test_autoencoder = loss_autoencoder(batch_parameter_pred_test_AE, batch_parameter_test)
            unscaled_replica_batch_loss_test_forward_problem = loss_forward_problem(batch_state_pred_test, batch_state_obs_test, hyperp.penalty)
            unscaled_replica_batch_loss_test = unscaled_replica_batch_loss_test_autoencoder + unscaled_replica_batch_loss_test_forward_problem
            mean_loss_test_autoencoder(unscaled_replica_batch_loss_test_autoencoder)
            mean_loss_test_forward_problem(unscaled_replica_batch_loss_test_forward_problem)
            mean_loss_test(unscaled_replica_batch_loss_test)
            mean_relative_error_parameter_autoencoder(relative_error(batch_parameter_pred_test_AE, batch_parameter_test))
            mean_relative_error_parameter_inverse_problem(relative_error(batch_parameter_pred_test_Inverse_problem, batch_parameter_test))
            mean_relative_error_state_obs(relative_error(batch_state_pred_test, batch_state_obs_test))
        
        @tf.function
        def dist_test_step(batch_parameter_test, batch_state_obs_test):
            return dist_strategy.experimental_run_v2(test_step, (batch_parameter_test, batch_state_obs_test))

###############################################################################
#                             Train Neural Network                            #
############################################################################### 
    print('Beginning Training')
    for epoch in range(hyperp.num_epochs):
        print('================================')
        print('            Epoch %d            ' %(epoch))
        print('================================')
        print(file_paths.filename)
        print('GPUs: ' + run_options.dist_which_gpus + '\n')
        print('Optimizing %d batches of size %d:' %(num_batches_train, hyperp.batch_size))
        start_time_epoch = time.time()
        batch_counter = 0
        total_loss_train = 0
        for batch_parameter_train, batch_state_obs_train in dist_parameter_and_state_obs_train:
            start_time_batch = time.time()
            batch_loss_train = dist_train_step(batch_parameter_train, batch_state_obs_train)
            total_loss_train += batch_loss_train
            elapsed_time_batch = time.time() - start_time_batch
            #=== Display Model Summary ===#
            if batch_counter == 0 and epoch == 0:
                NN.summary()
            if batch_counter  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))
            batch_counter += 1
        mean_loss_train = total_loss_train/batch_counter
        
        #=== Computing Validation Metrics ===#
        for batch_parameter_val, batch_state_obs_val in dist_parameter_and_state_obs_val:
            dist_val_step(batch_parameter_val, batch_state_obs_val)
                
        #=== Computing Test Metrics ===#
        for batch_parameter_test, batch_state_obs_test in dist_parameter_and_state_obs_test:
            dist_test_step(batch_parameter_test, batch_state_obs_test)
        
        #=== Track Training Metrics, Weights and Gradients ===#
        with summary_writer.as_default():
            tf.summary.scalar('loss_training', mean_loss_train, step=epoch)
            tf.summary.scalar('loss_training_autoencoder', mean_loss_train_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_training_forward_problem', mean_loss_train_forward_problem.result(), step=epoch)
            tf.summary.scalar('loss_val', mean_loss_val.result(), step=epoch)
            tf.summary.scalar('loss_val_autoencoder', mean_loss_val_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_val_forward_problem', mean_loss_val_forward_problem.result(), step=epoch)
            tf.summary.scalar('loss_test', mean_loss_test.result(), step=epoch)
            tf.summary.scalar('loss_test_autoencoder', mean_loss_test_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_test_forward_problem', mean_loss_test_forward_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_autoencoder', mean_relative_error_parameter_autoencoder.result(), step=epoch)
            tf.summary.scalar('relative_error_parameter_inverse_problem', mean_relative_error_parameter_inverse_problem.result(), step=epoch)
            tf.summary.scalar('relative_error_state_obs', mean_relative_error_state_obs.result(), step=epoch)            
                
        #=== Update Storage Arrays ===#
        storage_array_loss_train = np.append(storage_array_loss_train, mean_loss_train)
        storage_array_loss_train_autoencoder = np.append(storage_array_loss_train_autoencoder, mean_loss_train_autoencoder.result())
        storage_array_loss_train_forward_problem = np.append(storage_array_loss_train_forward_problem, mean_loss_train_forward_problem.result())
        storage_array_loss_val = np.append(storage_array_loss_val, mean_loss_val.result())
        storage_array_loss_val_autoencoder = np.append(storage_array_loss_val_autoencoder, mean_loss_val_autoencoder.result())
        storage_array_loss_val_forward_problem = np.append(storage_array_loss_val_forward_problem, mean_loss_val_forward_problem.result())
        storage_array_loss_test = np.append(storage_array_loss_test, mean_loss_test)
        storage_array_loss_test_autoencoder = np.append(storage_array_loss_test_autoencoder, mean_loss_test_autoencoder.result())
        storage_array_loss_test_forward_problem = np.append(storage_array_loss_test_forward_problem, mean_loss_test_forward_problem.result())
        storage_array_relative_error_parameter_autoencoder = np.append(storage_array_relative_error_parameter_autoencoder, mean_relative_error_parameter_autoencoder.result())
        storage_array_relative_error_parameter_inverse_problem = np.append(storage_array_relative_error_parameter_inverse_problem, mean_relative_error_parameter_inverse_problem.result())
        storage_array_relative_error_state_obs = np.append(storage_array_relative_error_state_obs, mean_relative_error_state_obs.result())
            
        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(mean_loss_train, mean_loss_train_autoencoder.result(), mean_loss_train_forward_problem.result()))
        print('Val Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(mean_loss_val.result(), mean_loss_val_autoencoder.result(), mean_loss_val_forward_problem.result()))
        print('Test Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(mean_loss_test.result(), mean_loss_test_autoencoder.result(), mean_loss_test_forward_problem.result()))
        print('Rel Errors: AE: %.3e, Inverse: %.3e, Forward: %.3e\n' %(mean_relative_error_parameter_autoencoder.result(), mean_relative_error_parameter_inverse_problem.result(), mean_relative_error_state_obs.result()))
        start_time_epoch = time.time()   
        
        #=== Resetting Metrics ===#
        mean_loss_train_autoencoder.reset_states()
        mean_loss_train_forward_problem.reset_states()
        mean_loss_val.reset_states()
        mean_loss_val_autoencoder.reset_states()
        mean_loss_val_forward_problem.reset_states()    
        mean_loss_test.reset_states()
        mean_loss_test_autoencoder.reset_states()
        mean_loss_test_forward_problem.reset_states()
        mean_relative_error_parameter_autoencoder.reset_states()
        mean_relative_error_parameter_inverse_problem.reset_states()
        mean_relative_error_state_obs.reset_states()
            
    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 
    
    return storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs 