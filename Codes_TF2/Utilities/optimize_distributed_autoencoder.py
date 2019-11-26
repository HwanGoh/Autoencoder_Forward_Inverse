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
    parameter_and_state_obs_train = dist_strategy.experimental_distribute_dataset(parameter_and_state_obs_train)

    with dist_strategy.scope():    
        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Define Metrics ===#        
        unscaled_loss_train_batch_average_autoencoder = tf.keras.metrics.Sum() 
        unscaled_loss_train_batch_average_forward_problem = tf.keras.metrics.Sum()
        
        unscaled_loss_val_batch_average = tf.keras.metrics.Sum()
        unscaled_loss_val_batch_average_autoencoder = tf.keras.metrics.Sum()
        unscaled_loss_val_batch_average_forward_problem = tf.keras.metrics.Sum()
        
        unscaled_loss_test_batch_average = tf.keras.metrics.Mean()
        unscaled_loss_test_batch_average_autoencoder = tf.keras.metrics.Sum()
        unscaled_loss_test_batch_average_forward_problem = tf.keras.metrics.Sum()    
        
        unscaled_relative_error_batch_average_parameter_autoencoder = tf.keras.metrics.Mean()
        unscaled_relative_error_batch_average_parameter_inverse_problem = tf.keras.metrics.Mean()
        unscaled_relative_error_batch_average_state_obs = tf.keras.metrics.Mean()
    
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
        def train_step(parameter_train, state_obs_train):
            with tf.GradientTape() as tape:
                parameter_pred_train_AE = NN(parameter_train)
                state_pred_train = NN.encoder(parameter_train)
                unscaled_loss_train_batch_autoencoder_replica = loss_autoencoder(parameter_pred_train_AE, parameter_train)
                unscaled_loss_train_batch_forward_problem_replica = loss_forward_problem(state_pred_train, state_obs_train, hyperp.penalty)
                unscaled_loss_train_batch_replica = unscaled_loss_train_batch_autoencoder_replica + unscaled_loss_train_batch_forward_problem_replica
                loss_train_batch = tf.reduce_sum(unscaled_loss_train_batch_replica * (1./GLOBAL_BATCH_SIZE))
            gradients = tape.gradient(loss_train_batch, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            unscaled_loss_train_batch_average_autoencoder(unscaled_loss_train_batch_autoencoder_replica)
            unscaled_loss_train_batch_average_forward_problem(unscaled_loss_train_batch_forward_problem_replica)
            return loss_train_batch
        
        @tf.function
        def dist_train_step(parameter_train, state_obs_train):
            per_replica_losses = dist_strategy.experimental_run_v2(train_step, args=(parameter_train, state_obs_train))
            return dist_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
                        
        #=== Validation Step ===#
        def val_step(parameter_val, state_obs_val, penalty):
            parameter_pred_val_batch_AE = NN(parameter_val)
            state_pred_val_batch = NN.encoder(parameter_val)
            unscaled_loss_val_batch_autoencoder = loss_autoencoder(parameter_pred_val_batch_AE, parameter_val)
            unscaled_loss_val_batch_forward_problem = loss_forward_problem(state_pred_val_batch, state_obs_val, penalty)
            unscaled_loss_val_batch = unscaled_loss_val_batch_autoencoder + unscaled_loss_val_batch_forward_problem
            unscaled_loss_val_batch_average_autoencoder(unscaled_loss_val_batch_autoencoder)
            unscaled_loss_val_batch_average_forward_problem(unscaled_loss_val_batch_forward_problem)
            unscaled_loss_val_batch_average(unscaled_loss_val_batch)
        
        @tf.function
        def dist_val_step(parameter_val, state_obs_val, penalty):
            return dist_strategy.experimental_run_v2(val_step, (parameter_val, state_obs_val, penalty))
        
        #=== Test Step ===#
        def test_step(parameter_test, state_obs_test, penalty):
            parameter_pred_test_batch_AE = NN(parameter_test)
            parameter_pred_test_batch_Inverse_problem = NN.decoder(state_obs_test)
            state_pred_test_batch = NN.encoder(parameter_test)
            unscaled_loss_test_batch_autoencoder = loss_autoencoder(parameter_pred_test_batch_AE, parameter_test)
            unscaled_loss_test_batch_forward_problem = loss_forward_problem(state_pred_test_batch, state_obs_test, penalty)
            unscaled_loss_test_batch = unscaled_loss_test_batch_autoencoder + unscaled_loss_test_batch_forward_problem
            unscaled_loss_test_batch_average_autoencoder(unscaled_loss_test_batch_autoencoder)
            unscaled_loss_test_batch_average_forward_problem(unscaled_loss_test_batch_forward_problem)
            unscaled_loss_test_batch_average(unscaled_loss_test_batch)
            unscaled_relative_error_batch_average_parameter_autoencoder(relative_error(parameter_pred_test_batch_AE, parameter_test))
            unscaled_relative_error_batch_average_parameter_inverse_problem(relative_error(parameter_pred_test_batch_Inverse_problem, parameter_test))
            unscaled_relative_error_batch_average_state_obs(relative_error(state_pred_test_batch, state_obs_test))
        
        @tf.function
        def dist_test_step(parameter_test, state_obs_test, penalty):
            return dist_strategy.experimental_run_v2(test_step, (parameter_test, state_obs_test, penalty))

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
        batch_counter = 0
        total_loss_train = 0
        for parameter_train, state_obs_train in parameter_and_state_obs_train:
            start_time_batch = time.time()
            dist_loss_train_batch = dist_train_step(parameter_train, state_obs_train)
            total_loss_train += dist_loss_train_batch
            elapsed_time_batch = time.time() - start_time_batch
            #=== Display Model Summary ===#
            if batch_counter == 0 and epoch == 0:
                NN.summary()
            if batch_counter  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))
            batch_counter += 1
        loss_train = total_loss_train/batch_counter
        loss_train_batch_average_autoencoder = unscaled_loss_train_batch_average_autoencoder.result()/batch_counter
        loss_train_batch_average_forward_problem = unscaled_loss_train_batch_average_forward_problem.result()/batch_counter
        
        #=== Computing Validation Metrics ===#
        batch_counter = 0
        for parameter_val, state_obs_val in parameter_and_state_obs_val:
            dist_val_step(parameter_val, state_obs_val, hyperp.penalty)
            batch_counter += 1
        loss_val_batch_average = unscaled_loss_val_batch_average.result()/batch_counter
        loss_val_batch_average_autoencoder = unscaled_loss_val_batch_average_autoencoder.result()/batch_counter
        loss_val_batch_average_forward_problem = unscaled_loss_val_batch_average_forward_problem.result()/batch_counter
        
        
        #=== Computing Test Metrics ===#
        batch_counter = 0
        for parameter_test, state_obs_test in parameter_and_state_obs_test:
            dist_test_step(parameter_test, state_obs_test, hyperp.penalty)
            batch_counter += 1
        loss_test_batch_average = unscaled_loss_test_batch_average.result()/batch_counter
        loss_test_batch_average_autoencoder = unscaled_loss_test_batch_average_autoencoder.result()/batch_counter
        loss_test_batch_average_forward_problem = unscaled_loss_test_batch_average_forward_problem.result()/batch_counter
        relative_error_batch_average_parameter_autoencoder = unscaled_relative_error_batch_average_parameter_autoencoder.result()
        relative_error_batch_average_parameter_inverse_problem = unscaled_relative_error_batch_average_parameter_inverse_problem.result()
        relative_error_batch_average_state_obs = unscaled_relative_error_batch_average_state_obs.result()
        
        #=== Track Training Metrics, Weights and Gradients ===#
        with summary_writer.as_default():
            tf.summary.scalar('loss_training', loss_train, step=epoch)
            tf.summary.scalar('loss_training_autoencoder', loss_train_batch_average_autoencoder, step=epoch)
            tf.summary.scalar('loss_training_forward_problem', loss_train_batch_average_forward_problem, step=epoch)
            tf.summary.scalar('loss_val', loss_val_batch_average, step=epoch)
            tf.summary.scalar('loss_val_autoencoder', loss_val_batch_average_autoencoder, step=epoch)
            tf.summary.scalar('loss_val_forward_problem', loss_val_batch_average_forward_problem, step=epoch)
            tf.summary.scalar('loss_test', loss_test_batch_average, step=epoch)
            tf.summary.scalar('loss_test_autoencoder', loss_test_batch_average_autoencoder, step=epoch)
            tf.summary.scalar('loss_test_forward_problem', loss_test_batch_average_forward_problem, step=epoch)
            tf.summary.scalar('relative_error_parameter_autoencoder', relative_error_batch_average_parameter_autoencoder, step=epoch)
            tf.summary.scalar('relative_error_parameter_inverse_problem', relative_error_batch_average_parameter_inverse_problem, step=epoch)
            tf.summary.scalar('relative_error_state_obs', relative_error_batch_average_state_obs, step=epoch)            
                
        #=== Update Storage Arrays ===#
        storage_array_loss_train = np.append(storage_array_loss_train, loss_train)
        storage_array_loss_train_autoencoder = np.append(storage_array_loss_train_autoencoder, loss_train_batch_average_autoencoder)
        storage_array_loss_train_forward_problem = np.append(storage_array_loss_train_forward_problem, loss_train_batch_average_forward_problem)
        storage_array_loss_val = np.append(storage_array_loss_val, loss_val_batch_average)
        storage_array_loss_val_autoencoder = np.append(storage_array_loss_val_autoencoder, loss_val_batch_average_autoencoder)
        storage_array_loss_val_forward_problem = np.append(storage_array_loss_val_forward_problem, loss_val_batch_average_forward_problem)
        storage_array_loss_test = np.append(storage_array_loss_test, loss_test_batch_average)
        storage_array_loss_test_autoencoder = np.append(storage_array_loss_test_autoencoder, loss_test_batch_average_autoencoder)
        storage_array_loss_test_forward_problem = np.append(storage_array_loss_test_forward_problem, loss_test_batch_average_forward_problem)
        storage_array_relative_error_parameter_autoencoder = np.append(storage_array_relative_error_parameter_autoencoder, relative_error_batch_average_parameter_autoencoder)
        storage_array_relative_error_parameter_inverse_problem = np.append(storage_array_relative_error_parameter_inverse_problem, relative_error_batch_average_parameter_inverse_problem)
        storage_array_relative_error_state_obs = np.append(storage_array_relative_error_state_obs, relative_error_batch_average_state_obs)
            
        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_train, loss_train_batch_average_autoencoder, loss_train_batch_average_forward_problem))
        print('Val Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_val_batch_average, loss_val_batch_average_autoencoder, loss_val_batch_average_forward_problem))
        print('Test Loss: Full: %.3e, Parameter: %.3e, State: %.3e' %(loss_test_batch_average, loss_test_batch_average_autoencoder, loss_test_batch_average_forward_problem))
        print('Rel Errors: AE: %.3e, Inverse: %.3e, Forward: %.3e\n' %(relative_error_batch_average_parameter_autoencoder, relative_error_batch_average_parameter_inverse_problem, relative_error_batch_average_state_obs))
        start_time_epoch = time.time()   
        
        #=== Resetting Metrics ===#
        unscaled_loss_train_batch_average_autoencoder.reset_states()
        unscaled_loss_train_batch_average_forward_problem.reset_states()
        unscaled_loss_val_batch_average.reset_states()
        unscaled_loss_val_batch_average_autoencoder.reset_states()
        unscaled_loss_val_batch_average_forward_problem.reset_states()    
        unscaled_loss_test_batch_average.reset_states()
        unscaled_loss_test_batch_average_autoencoder.reset_states()
        unscaled_loss_test_batch_average_forward_problem.reset_states()
        unscaled_relative_error_batch_average_parameter_autoencoder.reset_states()
        unscaled_relative_error_batch_average_parameter_inverse_problem.reset_states()
        unscaled_relative_error_batch_average_state_obs.reset_states()
            
    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    print('Final Model Saved') 
    
    return storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs 