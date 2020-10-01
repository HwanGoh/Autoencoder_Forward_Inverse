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
        loss_penalized_difference, kld_loss, relative_error,
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
    if not os.path.exists(filepaths.directory_trained_NN):
        os.makedirs(filepaths.directory_trained_NN)

    #=== Tensorboard ===# "tensorboard --logdir=Tensorboard"
    if os.path.exists(filepaths.directory_tensorboard):
        shutil.rmtree(filepaths.directory_tensorboard)
    summary_writer = tf.summary.create_file_writer(filepaths.directory_tensorboard)

    #=== Display Neural Network Architecture ===#
    with dist_strategy.scope():
        NN.build((hyperp.batch_size, input_dimensions))
        NN.summary()

    #=== Setting Initial kld Penalty to be Incremented ===#
    penalty_kld = 0

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    with dist_strategy.scope():
        #=== Training Step ===#
        def train_step(batch_input_train, batch_latent_train, penalty_kld):
            with tf.GradientTape() as tape:
                batch_likelihood_train = NN(batch_input_train)
                batch_post_mean_train, batch_log_post_var_train = NN.encoder(batch_input_train)

                unscaled_replica_batch_loss_train_vae =\
                        loss_weighted_penalized_difference(
                                batch_input_train, batch_likelihood_train,
                                noise_regularization_matrix, 1)
                unscaled_replica_batch_loss_train_kld = kld_loss(
                        batch_post_mean_train, batch_log_post_var_train,
                        prior_mean, prior_cov_inv,
                        log_det_prior_cov, latent_dimension,
                        penalty_kld)
                unscaled_replica_batch_loss_train_post_draw = loss_penalized_difference(
                        batch_latent_train,
                        NN.reparameterize(batch_post_mean_train, batch_log_post_var_train),
                        hyperp.penalty_post_draw)

                unscaled_replica_batch_loss_train =\
                        -(-unscaled_replica_batch_loss_train_vae\
                          -unscaled_replica_batch_loss_train_kld\
                          -unscaled_replica_batch_loss_train_post_draw)
                scaled_replica_batch_loss_train = tf.reduce_sum(
                        unscaled_replica_batch_loss_train * (1./hyperp.batch_size))

            gradients = tape.gradient(scaled_replica_batch_loss_train, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            metrics.mean_loss_train_vae(unscaled_replica_batch_loss_train_vae)
            metrics.mean_loss_train_encoder(unscaled_replica_batch_loss_train_kld)
            metrics.mean_loss_train_post_draw(unscaled_replica_batch_loss_train_post_draw)

            return scaled_replica_batch_loss_train

        @tf.function
        def dist_train_step(batch_input_train, batch_latent_train, penalty_kld):
            per_replica_losses = dist_strategy.experimental_run_v2(
                    train_step, args=(batch_input_train, batch_latent_train, penalty_kld))
            return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        #=== Validation Step ===#
        def val_step(batch_input_val, batch_latent_val, penalty_kld):
            batch_likelihood_val = NN(batch_input_val)
            batch_post_mean_val, batch_log_post_var_val = NN.encoder(batch_input_val)

            unscaled_replica_batch_loss_val_vae = loss_weighted_penalized_difference(
                    batch_input_val, batch_likelihood_val,
                    noise_regularization_matrix, 1)
            unscaled_replica_batch_loss_val_kld = kld_loss(
                    batch_post_mean_val, batch_log_post_var_val,
                    prior_mean, prior_cov_inv,
                    log_det_prior_cov, latent_dimension,
                    penalty_kld)
            unscaled_replica_batch_loss_val_post_draw = loss_penalized_difference(
                    batch_latent_val,
                    NN.reparameterize(batch_post_mean_val, batch_log_post_var_val),
                    hyperp.penalty_post_draw)

            unscaled_replica_batch_loss_val =\
                    -(-unscaled_replica_batch_loss_val_vae\
                      -unscaled_replica_batch_loss_val_kld\
                      -unscaled_replica_batch_loss_val_post_draw)

            metrics.mean_loss_val(unscaled_replica_batch_loss_val)
            metrics.mean_loss_val_vae(unscaled_replica_batch_loss_val_vae)
            metrics.mean_loss_val_encoder(unscaled_replica_batch_loss_val_kld)
            metrics.mean_loss_val_post_draw(unscaled_replica_batch_loss_val_post_draw)

        # @tf.function
        def dist_val_step(batch_input_val, batch_latent_val, penalty_kld):
            return dist_strategy.experimental_run_v2(
                    val_step, (batch_input_val, batch_latent_val, penalty_kld))

        #=== Test Step ===#
        def test_step(batch_input_test, batch_latent_test, penalty_kld):
            batch_likelihood_test = NN(batch_input_test)
            batch_post_mean_test, batch_log_post_var_test = NN.encoder(batch_input_test)
            batch_input_pred_test = NN.decoder(batch_latent_test)

            unscaled_replica_batch_loss_test_vae =\
                    loss_weighted_penalized_difference(
                            batch_input_test, batch_likelihood_test,
                            noise_regularization_matrix, 1)
            unscaled_replica_batch_loss_test_kld = kld_loss(
                    batch_post_mean_test, batch_log_post_var_test,
                    prior_mean, prior_cov_inv, log_det_prior_cov, latent_dimension,
                    penalty_kld)
            unscaled_replica_batch_loss_test_post_draw = loss_penalized_difference(
                    batch_latent_test,
                    NN.reparameterize(batch_post_mean_test, batch_log_post_var_test),
                    hyperp.penalty_post_draw)

            unscaled_replica_batch_loss_test =\
                    -(-unscaled_replica_batch_loss_test_vae\
                      -unscaled_replica_batch_loss_test_kld\
                      -unscaled_replica_batch_loss_val_post_draw)

            metrics.mean_loss_test(unscaled_replica_batch_loss_test)
            metrics.mean_loss_test_vae(unscaled_replica_batch_loss_test_vae)
            metrics.mean_loss_test_encoder(unscaled_replica_batch_loss_test_kld)
            metrics.mean_loss_test_post_draw(unscaled_replica_batch_loss_test_post_draw)

            metrics.mean_relative_error_input_vae(relative_error(
                batch_input_test, batch_input_likelihood_test))
            metrics.mean_relative_error_latent_post_draw(relative_error(
                batch_latent_test, NN.reparameterize(batch_post_mean_test, batch_log_post_var_test)))
            metrics.mean_relative_error_input_decoder(relative_error(
                batch_input_test, batch_input_pred_test))

        # @tf.function
        def dist_test_step(batch_input_test, batch_latent_test, penalty_kld):
            return dist_strategy.experimental_run_v2(
                    test_step, (batch_input_test, batch_latent_test, penalty_kld))

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
            batch_loss_train = dist_train_step(
                    batch_input_train, batch_latent_train, penalty_kld)
            total_loss_train += batch_loss_train
            elapsed_time_batch = time.time() - start_time_batch
            if batch_counter  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))
            batch_counter += 1
        metrics.mean_loss_train = total_loss_train/batch_counter

        #=== Computing Validation Metrics ===#
        for batch_input_val, batch_latent_val in dist_input_and_latent_val:
            dist_val_step(batch_input_val, batch_latent_val, penalty_kld)

        #=== Computing Test Metrics ===#
        for batch_input_test, batch_latent_test in dist_input_and_latent_test:
            dist_test_step(batch_input_test, batch_latent_test, penalty_kld)

        #=== Tensorboard Tracking Training Metrics, Weights and Gradients ===#
        metrics.update_tensorboard(summary_writer, epoch)

        #=== Update Storage Arrays ===#
        metrics.update_storage_arrays()

        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, VAE: %.3e, kld: %.3e, post_draw: %.3e'\
                %(metrics.mean_loss_train,
                  metrics.mean_loss_train_vae.result(),
                  metrics.mean_loss_train_encoder.result(),
                  metrics.mean_loss_train_post_draw.result()))
        print('Val Loss: Full: %.3e, VAE: %.3e, kld: %.3e, post_draw: %.3e'\
                %(metrics.mean_loss_val.result(),
                  metrics.mean_loss_val_vae.result(),
                  metrics.mean_loss_val_encoder.result(),
                  metrics.mean_loss_val_post_draw.result()))
        print('Test Loss: Full: %.3e, VAE: %.3e, kld: %.3e, post_draw: %.3e'\
                %(metrics.mean_loss_test.result(),
                  metrics.mean_loss_test_vae.result(),
                  metrics.mean_loss_test_encoder.result(),
                  metrics.mean_loss_val_post_draw.result()))
        print('Rel Errors: VAE: %.3e, Post Draw: %.3e, Decoder: %.3e\n'\
                %(metrics.mean_relative_error_input_vae.result(),
                  metrics.mean_relative_error_latent_post_draw.result(),
                  metrics.mean_relative_error_input_decoder.result()))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Save Current Model and Metrics ===#
        if epoch % 5 == 0:
            NN.save_weights(filepaths.trained_NN)
            metrics.save_metrics(filepaths)
            dump_attrdict_as_yaml(hyperp, filepaths.directory_trained_NN, 'hyperp')
            dump_attrdict_as_yaml(options, filepaths.directory_trained_NN, 'options')
            print('Current Model and Metrics Saved')

        #=== Increase kld Penalty ===#
        if epoch %hyperp.penalty_kld_rate == 0 and epoch != 0:
            penalty_kld += hyperp.penalty_kld_incr

    #=== Save Final Model ===#
    NN.save_weights(filepaths.trained_NN)
    metrics.save_metrics(filepaths)
    dump_attrdict_as_yaml(hyperp, filepaths.directory_trained_NN, 'hyperp')
    dump_attrdict_as_yaml(options, filepaths.directory_trained_NN, 'options')
    print('Final Model and Metrics Saved')
