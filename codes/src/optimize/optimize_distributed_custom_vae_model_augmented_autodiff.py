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

from utils_training.metrics_distributed_vae import Metrics
from utils_io.config_io import dump_attrdict_as_yaml

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_distributed(dist_strategy,
        hyperp, options, filepaths,
        NN, optimizer,
        loss_penalized_difference, KLD_loss, relative_error,
        prior_mean, prior_covariance,
        input_and_latent_train, input_and_latent_val, input_and_latent_test,
        input_dimensions, latent_dimension,
        num_batches_train,
        loss_weighted_penalized_difference, noise_regularization_matrix,
        positivity_constraint):

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
    if not os.path.exists(filepaths.NN_savefile_directory):
        os.makedirs(filepaths.NN_savefile_directory)

    #=== Tensorboard ===# "tensorboard --logdir=Tensorboard"
    if os.path.exists(filepaths.tensorboard_directory):
        shutil.rmtree(filepaths.tensorboard_directory)
    summary_writer = tf.summary.create_file_writer(filepaths.tensorboard_directory)

    #=== Display Neural Network Architecture ===#
    with dist_strategy.scope():
        NN.build((hyperp.batch_size, input_dimensions))
        NN.summary()

    #=== Setting Initial KLD Penalty to be Incremented ===#
    penalty_KLD = 0

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    with dist_strategy.scope():
        #=== Training Step ===#
        def train_step(batch_input_train, batch_latent_train):
            with tf.GradientTape() as tape:
                batch_post_mean_train, batch_log_post_var_train = NN.encoder(batch_input_train)
                batch_input_pred_forward_model_train = solve_forward_model(positivity_constraint(
                    NN.reparameterize(batch_post_mean_train, batch_log_post_var_train)))

                unscaled_replica_batch_loss_train_VAE =\
                        loss_weighted_penalized_difference(
                                batch_input_train, batch_input_pred_forward_model_train,
                                noise_regularization_matrix, 1)
                unscaled_replica_batch_loss_loss_train_KLD = KLD_loss(
                        batch_post_mean_train, batch_log_post_var_train,
                        batch_latent_train, prior_cov_inv,
                        log_det_prior_cov, latent_dimension,
                        penalty_KLD)
                unscaled_replica_batch_loss_train_post_mean = loss_penalized_difference(
                        batch_latent_train, batch_post_mean_train,
                        hyperp.penalty_post_mean)

                unscaled_replica_batch_loss_train =\
                        -(-unscaled_replica_batch_loss_train_VAE\
                          -unscaled_replica_batch_loss_loss_train_KLD\
                          -unscaled_replica_batch_loss_train_post_mean)
                scaled_replica_batch_loss_train = tf.reduce_sum(
                        unscaled_replica_batch_loss_train * (1./hyperp.batch_size))

            gradients = tape.gradient(scaled_replica_batch_loss_train, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            metrics.mean_loss_train_VAE(-unscaled_replica_batch_loss_train_VAE)
            metrics.mean_loss_train_encoder(unscaled_replica_batch_loss_loss_train_KLD)
            metrics.mean_loss_train_post_mean(unscaled_replica_batch_loss_train_post_mean)

            return scaled_replica_batch_loss_train

        # @tf.function
        def dist_train_step(batch_input_train, batch_latent_train):
            per_replica_losses = dist_strategy.experimental_run_v2(
                    train_step, args=(batch_input_train, batch_latent_train))
            return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        #=== Validation Step ===#
        def val_step(batch_input_val, batch_latent_val):
            batch_post_mean_val, batch_log_post_var_val = NN.encoder(batch_input_val)

            unscaled_replica_batch_loss_val_KLD = KLD_loss(
                    batch_post_mean_val, batch_log_post_var_val,
                    batch_latent_val, prior_cov_inv,
                    log_det_prior_cov, latent_dimension,
                    penalty_KLD)
            unscaled_replica_batch_loss_val_post_mean = loss_penalized_difference(
                    batch_latent_val, batch_post_mean_val,
                    hyperp.penalty_post_mean)

            unscaled_replica_batch_loss_val =\
                    -(-unscaled_replica_batch_loss_val_KLD\
                      -unscaled_replica_batch_loss_val_post_mean)

            metrics.mean_loss_val(unscaled_replica_batch_loss_val)
            metrics.mean_loss_val_encoder(unscaled_replica_batch_loss_val_KLD)
            metrics.mean_loss_val_post_mean(unscaled_replica_batch_loss_val_post_mean)

        # @tf.function
        def dist_val_step(batch_input_val, batch_latent_val):
            return dist_strategy.experimental_run_v2(val_step, (batch_input_val, batch_latent_val))

        #=== Test Step ===#
        def test_step(batch_input_test, batch_latent_test):
            batch_post_mean_test, batch_log_post_var_test = NN.encoder(batch_input_test)

            unscaled_replica_batch_loss_test_KLD = KLD_loss(
                    batch_post_mean_test, batch_log_post_var_test,
                    batch_latent_test, prior_cov_inv, log_det_prior_cov, latent_dimension,
                    penalty_KLD)
            unscaled_replica_batch_loss_test_post_mean = loss_penalized_difference(
                    batch_latent_test, batch_post_mean_test,
                    hyperp.penalty_post_mean)

            unscaled_replica_batch_loss_test =\
                    -(-unscaled_replica_batch_loss_test_KLD\
                      -unscaled_replica_batch_loss_val_post_mean)

            metrics.mean_loss_test(unscaled_replica_batch_loss_test)
            metrics.mean_loss_test_encoder(unscaled_replica_batch_loss_test_KLD)
            metrics.mean_loss_test_post_mean(unscaled_replica_batch_loss_test_post_mean)

            metrics.mean_relative_error_latent_encoder(relative_error(
                batch_latent_test, batch_post_mean_test))

        # @tf.function
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
        print('Case: ' + filepaths.case_name + '\n' + 'NN: ' + filepaths.NN_name + '\n')
        print('GPUs: ' + options.dist_which_gpus + '\n')
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
        print('Train Loss: Full: %.3e, VAE: %.3e, KLD: %.3e, post_mean: %.3e'\
                %(metrics.mean_loss_train,
                  metrics.mean_loss_train_VAE.result(),
                  metrics.mean_loss_train_encoder.result(),
                  metrics.mean_loss_train_post_mean.result()))
        print('Val Loss: Full: %.3e, KLD: %.3e, post_mean: %.3e'\
                %(metrics.mean_loss_val.result(),
                  metrics.mean_loss_val_encoder.result(),
                  metrics.mean_loss_val_post_mean.result()))
        print('Test Loss: Full: %.3e, KLD: %.3e, post_mean: %.3e'\
                %(metrics.mean_loss_test.result(),
                  metrics.mean_loss_test_encoder.result(),
                  metrics.mean_loss_val_post_mean.result()))
        print('Rel Errors: Encoder: %.3e\n'\
                %(metrics.mean_relative_error_latent_encoder.result()))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Save Current Model and Metrics ===#
        if epoch % 5 == 0:
            NN.save_weights(filepaths.NN_savefile_name)
            metrics.save_metrics(filepaths)
            dump_attrdict_as_yaml(hyperp, filepaths.NN_savefile_directory, 'hyperp')
            dump_attrdict_as_yaml(options, filepaths.NN_savefile_directory, 'options')
            print('Current Model and Metrics Saved')

        #=== Increase KLD Penalty ===#
        if epoch %hyperp.penalty_KLD_rate == 0 and epoch != 0:
            penalty_KLD += hyperp.penalty_KLD_incr

    #=== Save Final Model ===#
    NN.save_weights(filepaths.NN_savefile_name)
    metrics.save_metrics(filepaths)
    dump_attrdict_as_yaml(hyperp, filepaths.NN_savefile_directory, 'hyperp')
    dump_attrdict_as_yaml(options, filepaths.NN_savefile_directory, 'options')
    print('Final Model Saved')
