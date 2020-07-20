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

from metrics_distributed_AE_model_aware import Metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE,
        hyperp, run_options, file_paths,
        NN, optimizer,
        loss_penalized_difference, KLD_loss, relative_error,
        prior_mean, prior_covariance,
        input_and_latent_train, input_and_latent_val, input_and_latent_test,
        input_input_shape, latent_dimension,
        num_batches_train):

    #=== Matrix Determinants and Inverse of Prior Covariance ===#
    prior_cov_inv = np.linalg.inv(prior_covariance)
    (sign, logdet) = np.linalg.slogdet(prior_covariance)
    log_det_prior_cov = sign*logdet

    #=== Check Number of Parallel Computations and Set Global Batch Size ===#
    print('Number of Replicas in Sync: %d' %(dist_strategy.num_replicas_in_sync))

    #=== Distribute Data ===#
    dist_input_and_latent_train =\
            dist_strategy.experimental_distribute_dataset(input_and_latent_train)
    dist_input_and_latent_val = dist_strategy.experimental_distribute_dataset(input_and_latent_val)
    dist_input_and_latent_test = dist_strategy.experimental_distribute_dataset(input_and_latent_test)

    #=== Metrics ===#
    metrics = Metrics(dist_strategy)

    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)

    #=== Tensorboard ===# "tensorboard --logdir=Tensorboard"
    if os.path.exists(file_paths.tensorboard_directory):
        shutil.rmtree(file_paths.tensorboard_directory)
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

    #=== Display Neural Network Architecture ===#
    with dist_strategy.scope():
        NN.build((hyperp.batch_size, input_dimensions))
        NN.summary()

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    with dist_strategy.scope():
        #=== Training Step ===#
        def train_step(batch_input_train, batch_latent_train):
            with tf.GradientTape() as tape:
                batch_likelihood_train = NN(batch_input_train)
                batch_post_mean_train, batch_log_post_var_train = NN.encoder(batch_input_train)

                unscaled_replica_batch_loss_train_VAE =\
                        loss_penalized_difference(
                                batch_likelihood_train, batch_input_train, 1)
                unscaled_replica_batch_loss_loss_train_KLD = KLD_loss(
                        batch_post_mean_train, batch_log_post_var_train,
                        prior_mean, prior_cov_inv,
                        log_det_prior_cov, latent_dimension)

                unscaled_replica_batch_loss_train =\
                        -(unscaled_replica_batch_loss_train_VAE
                                -unscaled_replica_batch_loss_loss_train_KLD)
                scaled_replica_batch_loss_train = tf.reduce_sum(
                        unscaled_replica_batch_loss_train * (1./GLOBAL_BATCH_SIZE))

            gradients = tape.gradient(scaled_replica_batch_loss_train, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            metrics.mean_loss_train_autoencoder(-unscaled_replica_batch_loss_train_VAE)
            metrics.mean_loss_train_encoder(unscaled_replica_batch_loss_loss_train_KLD)
            return scaled_replica_batch_loss_train

        @tf.function
        def dist_train_step(batch_input_train, batch_latent_train):
            per_replica_losses = dist_strategy.experimental_run_v2(
                    train_step, args=(batch_input_train, batch_latent_train))
            return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        #=== Validation Step ===#
        def val_step(batch_input_val, batch_latent_val):
            batch_likelihood_val = NN(batch_input_val)
            batch_post_mean_val, batch_log_post_var_val = NN.encoder(batch_input_val)

            unscaled_replica_batch_loss_val_VAE = loss_penalized_difference(
                    batch_likelihood_val, batch_input_val, 1)
            unscaled_replica_batch_loss_val_KLD = KLD_loss(
                    batch_post_mean_val, batch_log_post_var_val,
                    prior_mean, prior_cov_inv,
                    log_det_prior_cov, latent_dimension)
            unscaled_replica_batch_loss_val =\
                    -(unscaled_replica_batch_loss_val_VAE - unscaled_replica_batch_loss_val_KLD)

            metrics.mean_loss_val_autoencoder(-unscaled_replica_batch_loss_val_VAE)
            metrics.mean_loss_val_encoder(unscaled_replica_batch_loss_val_KLD)
            metrics.mean_loss_val(unscaled_replica_batch_loss_val)

        @tf.function
        def dist_val_step(batch_input_val, batch_latent_val):
            return dist_strategy.experimental_run_v2(val_step, (batch_input_val, batch_latent_val))

        #=== Test Step ===#
        def test_step(batch_input_test, batch_latent_test):
            batch_input_likelihood_test = NN(batch_input_test)
            batch_post_mean_test, batch_log_post_var_test = NN.encoder(batch_input_test)
            batch_input_pred_test = NN.decoder(batch_latent_test)

            unscaled_replica_batch_loss_test_VAE =\
                    loss_penalized_difference(
                            batch_input_likelihood_test, batch_input_test, 1)
            unscaled_replica_batch_loss_test_KLD = KLD_loss(
                    batch_post_mean_test, batch_log_post_var_test,
                    prior_mean, prior_cov_inv, log_det_prior_cov, latent_dimension)
            unscaled_replica_batch_loss_test =\
                    -(unscaled_replica_batch_loss_test_VAE - unscaled_replica_batch_loss_test_KLD)

            metrics.mean_loss_test_autoencoder(-unscaled_replica_batch_loss_test_VAE)
            metrics.mean_loss_test_encoder(unscaled_replica_batch_loss_test_KLD)
            metrics.mean_loss_test(unscaled_replica_batch_loss_test)

            metrics.mean_relative_error_input_autoencoder(relative_error(
                batch_input_test, batch_input_likelihood_test))
            metrics.mean_relative_error_latent_encoder(relative_error(
                batch_latent_test, batch_post_mean_test))
            metrics.mean_relative_error_input_decoder(relative_error(
                batch_input_test, batch_input_pred_test))

        @tf.function
        def dist_test_step(batch_input_test, batch_latent_test):
            return dist_strategy.experimental_run_v2(test_step, (batch_input_test,
                batch_latent_test))

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
        for batch_input_train, batch_latent_train in dist_input_and_latent_train:
            start_time_batch = time.time()
            #=== Compute Train Step ===#
            batch_loss_train = dist_train_step(batch_input_train, batch_latent_train)
            total_loss_train += batch_loss_train
            elapsed_time_batch = time.time() - start_time_batch
            if batch_counter  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))
            batch_counter += 1
        mean_loss_train = total_loss_train/batch_counter

        #=== Computing Validation Metrics ===#
        for batch_input_val, batch_latent_val in dist_input_and_latent_val:
            dist_val_step(batch_input_val, batch_latent_val)

        #=== Computing Test Metrics ===#
        for batch_input_test, batch_latent_test in dist_input_and_latent_test:
            dist_test_step(batch_input_test, batch_latent_test)

        #=== Tensorboard Tracking Training Metrics, Weights and Gradients ===#
        metrics.update_tensorboard(summary_writer, epoch)

        #=== Update Storage Arrays ===#
        metrics.update_storage_arrays()

        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_train,
                    metrics.mean_loss_train_autoencoder.result(),
                    metrics.mean_loss_train_encoder.result()))
        print('Val Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_val.result(),
                    metrics.mean_loss_val_autoencoder.result(),
                    metrics.mean_loss_val_encoder.result()))
        print('Test Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_test.result(),
                    metrics.mean_loss_test_autoencoder.result(),
                    metrics.mean_loss_test_encoder.result()))
        print('Rel Errors: AE: %.3e, Encoder: %.3e, Decoder: %.3e\n'\
                %(metrics.mean_relative_error_input_autoencoder.result(),
                    metrics.mean_relative_error_latent_encoder.result(),
                    metrics.mean_relative_error_input_decoder.result()))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Save Current Model and Metrics ===#
        if epoch % 5 == 0:
            NN.save_weights(file_paths.NN_savefile_name)
            metrics.save_metrics(file_paths)
            print('Current Model and Metrics Saved')

    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    metrics.save_metrics(file_paths)
    print('Final Model Saved')