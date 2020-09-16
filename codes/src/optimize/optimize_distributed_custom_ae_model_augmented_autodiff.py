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

from utils_training.metrics_distributed_ae import Metrics
from utils_config.config_io import dump_attrdict_as_yaml

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize_distributed(dist_strategy,
        hyperp, options, file_paths,
        NN, optimizer,
        loss_penalized_difference, relative_error,
        input_and_latent_train, input_and_latent_val, input_and_latent_test,
        input_dimensions,
        num_batches_train,
        loss_weighted_penalized_difference, noise_regularization_matrix,
        reg_prior, prior_mean, prior_covariance_cholesky_inverse,
        positivity_constraint,
        solve_forward_model):

    #=== Check Number of Parallel Computations and Set Global Batch Size ===#
    print('Number of Replicas in Sync: %d' %(dist_strategy.num_replicas_in_sync))

    #=== Distribute Data ===#
    dist_input_and_latent_train =\
            dist_strategy.experimental_distribute_dataset(input_and_latent_train)
    dist_input_and_latent_val =\
            dist_strategy.experimental_distribute_dataset(input_and_latent_val)
    dist_input_and_latent_test =\
            dist_strategy.experimental_distribute_dataset(input_and_latent_test)

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
                batch_input_pred_train_AE = NN(batch_input_train)
                batch_latent_pred_train = NN.encoder(batch_input_train)
                batch_input_pred_train = NN.decoder(batch_latent_train)

                if options.standard_autoencoder == 1:
                    batch_input_pred_forward_model_train =\
                            solve_forward_model(batch_input_pred_train_AE)

                    unscaled_replica_batch_loss_train_autoencoder =\
                            loss_penalized_difference(
                                    batch_input_train, batch_input_pred_train_AE, 1)
                    unscaled_replica_batch_loss_train_encoder =\
                            loss_weighted_penalized_difference(
                                    batch_latent_train, batch_latent_pred_train,
                                    noise_regularization_matrix, hyperp.penalty_encoder)
                    unscaled_replica_batch_loss_train_decoder =\
                            loss_penalized_difference(
                                    batch_input_train, batch_input_pred_train,
                                    hyperp.penalty_decoder)
                    unscaled_replica_batch_reg_train_prior = reg_prior(
                            batch_input_pred_train_AE,
                            prior_mean, prior_covariance_cholesky_inverse,
                            hyperp.penalty_prior)
                    unscaled_replica_batch_loss_train_forward_model =\
                            loss_weighted_penalized_difference(
                                    batch_latent_train, batch_input_pred_forward_model_train,
                                    noise_regularization_matrix, hyperp.penalty_aug)

                if options.reverse_autoencoder == 1:
                    unscaled_replica_batch_loss_train_autoencoder =\
                            loss_weighted_penalized_difference(
                                    batch_input_train,
                                    batch_input_pred_train_AE,
                                    noise_regularization_matrix, 1)
                    unscaled_replica_batch_loss_train_encoder =\
                            loss_penalized_difference(
                                    batch_latent_train,
                                    batch_latent_pred_train,
                                    hyperp.penalty_encoder)
                    unscaled_replica_batch_loss_train_decoder =\
                            loss_weighted_penalized_difference(
                                    batch_input_train, batch_input_pred_train,
                                    noise_regularization_matrix,
                                    hyperp.penalty_decoder)
                    unscaled_replica_batch_reg_train_prior = reg_prior(
                            batch_latent_pred_train,
                            prior_mean, prior_covariance_cholesky_inverse,
                            hyperp.penalty_prior)
                    unscaled_replica_batch_loss_train_forward_model =\
                            loss_weighted_penalized_difference(
                                    batch_input_train, batch_latent_pred_forward_model_train,
                                    noise_regularization_matrix, hyperp.penalty_aug)

                unscaled_replica_batch_loss_train =\
                        unscaled_replica_batch_loss_train_autoencoder +\
                        unscaled_replica_batch_loss_train_encoder +\
                        unscaled_replica_batch_loss_train_decoder +\
                        unscaled_replica_reg_train_prior
                scaled_replica_batch_loss_train = tf.reduce_sum(
                        unscaled_replica_batch_loss_train * (1./hyperp.batch_size))

            gradients = tape.gradient(scaled_replica_batch_loss_train, NN.trainable_variables)
            optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
            metrics.mean_loss_train_autoencoder(unscaled_replica_batch_loss_train_autoencoder)
            metrics.mean_loss_train_encoder(unscaled_replica_batch_loss_train_encoder)
            metrics.mean_loss_train_decoder(unscaled_replica_batch_loss_train_decoder)
            return scaled_replica_batch_loss_train

        @tf.function
        def dist_train_step(batch_input_train, batch_latent_train):
            per_replica_losses = dist_strategy.experimental_run_v2(
                    train_step, args=(batch_input_train, batch_latent_train))
            return dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        #=== Validation Step ===#
        def val_step(batch_input_val, batch_latent_val):
            batch_input_pred_val_AE = NN(batch_input_val)
            batch_latent_pred_val = NN.encoder(batch_input_val)
            batch_input_pred_val = NN.decoder(batch_latent_val)

            if options.standard_autoencoder == 1:
                unscaled_replica_batch_loss_val_autoencoder =\
                        loss_penalized_difference(
                                batch_input_val, batch_input_pred_val_AE, 1)
                unscaled_replica_batch_loss_val_encoder =\
                        loss_weighted_penalized_difference(
                                batch_latent_val, batch_latent_pred_val,
                                noise_regularization_matrix, hyperp.penalty_encoder)
                unscaled_replica_batch_loss_val_decoder =\
                        loss_penalized_difference(
                                batch_input_val, batch_input_pred_val,
                                hyperp.penalty_decoder)
                unscaled_replica_batch_reg_val_prior = reg_prior(
                        batch_input_pred_val_AE,
                        prior_mean, prior_covariance_cholesky_inverse,
                        hyperp.penalty_prior)

            if options.reverse_autoencoder == 1:
                unscaled_replica_batch_loss_val_autoencoder =\
                        loss_weighted_penalized_difference(
                                batch_input_val,
                                batch_input_pred_val_AE,
                                noise_regularization_matrix, 1)
                unscaled_replica_batch_loss_val_encoder =\
                        loss_penalized_difference(
                                batch_latent_val,
                                batch_latent_pred_val,
                                hyperp.penalty_encoder)
                unscaled_replica_batch_loss_val_decoder =\
                        loss_weighted_penalized_difference(
                                batch_input_val, batch_input_pred_val,
                                noise_regularization_matrix,
                                hyperp.penalty_decoder)
                unscaled_replica_batch_reg_val_prior = reg_prior(
                        batch_latent_pred_val,
                        prior_mean, prior_covariance_cholesky_inverse,
                        hyperp.penalty_prior)

                unscaled_replica_batch_loss_val =\
                        unscaled_replica_batch_loss_val_autoencoder +\
                        unscaled_replica_batch_loss_val_encoder +\
                        unscaled_replica_batch_loss_val_decoder +\
                        unscaled_replica_reg_val_prior
                scaled_replica_batch_loss_val = tf.reduce_sum(
                        unscaled_replica_batch_loss_val * (1./hyperp.batch_size))

            metrics.mean_loss_val(unscaled_replica_batch_loss_val)
            metrics.mean_loss_val_autoencoder(unscaled_replica_batch_loss_val_autoencoder)
            metrics.mean_loss_val_encoder(unscaled_replica_batch_loss_val_encoder)
            metrics.mean_loss_val_decoder(unscaled_replica_batch_loss_val_decoder)

        @tf.function
        def dist_val_step(batch_input_val, batch_latent_val):
            return dist_strategy.experimental_run_v2(val_step, (batch_input_val, batch_latent_val))

        #=== Test Step ===#
        def test_step(batch_input_test, batch_latent_test):
            batch_input_pred_test_AE = NN(batch_input_test)
            batch_latent_pred_test = NN.encoder(batch_input_test)
            batch_input_pred_test = NN.decoder(batch_latent_test)

            if options.standard_autoencoder == 1:
                unscaled_replica_batch_loss_test_autoencoder =\
                        loss_penalized_difference(
                                batch_input_test, batch_input_pred_test_AE, 1)
                unscaled_replica_batch_loss_test_encoder =\
                        loss_weighted_penalized_difference(
                                batch_latent_test, batch_latent_pred_test,
                                noise_regularization_matrix, hyperp.penalty_encoder)
                unscaled_replica_batch_loss_test_decoder =\
                        loss_penalized_difference(
                                batch_input_test, batch_input_pred_test,
                                hyperp.penalty_decoder)
                unscaled_replica_batch_reg_test_prior = reg_prior(
                        batch_input_pred_test_AE,
                        prior_mean, prior_covariance_cholesky_inverse,
                        hyperp.penalty_prior)

            if options.reverse_autoencoder == 1:
                unscaled_replica_batch_loss_test_autoencoder =\
                        loss_weighted_penalized_difference(
                                batch_input_test,
                                batch_input_pred_test_AE,
                                noise_regularization_matrix, 1)
                unscaled_replica_batch_loss_test_encoder =\
                        loss_penalized_difference(
                                batch_latent_test,
                                batch_latent_pred_test,
                                hyperp.penalty_encoder)
                unscaled_replica_batch_loss_test_decoder =\
                        loss_weighted_penalized_difference(
                                batch_input_test, batch_input_pred_test,
                                noise_regularization_matrix,
                                hyperp.penalty_decoder)
                unscaled_replica_batch_reg_test_prior = reg_prior(
                        batch_latent_pred_test,
                        prior_mean, prior_covariance_cholesky_inverse,
                        hyperp.penalty_prior)

                unscaled_replica_batch_loss_test =\
                        unscaled_replica_batch_loss_test_autoencoder +\
                        unscaled_replica_batch_loss_test_encoder +\
                        unscaled_replica_batch_loss_test_decoder +\
                        unscaled_replica_reg_test_prior
                scaled_replica_batch_loss_test = tf.reduce_sum(
                        unscaled_replica_batch_loss_test * (1./hyperp.batch_size))

            metrics.mean_loss_test(unscaled_replica_batch_loss_test)
            metrics.mean_loss_test_autoencoder(unscaled_replica_batch_loss_test_autoencoder)
            metrics.mean_loss_test_encoder(unscaled_replica_batch_loss_test_encoder)
            metrics.mean_loss_test_decoder(unscaled_replica_batch_loss_test_decoder)

            metrics.mean_relative_error_data_autoencoder(
                    relative_error(batch_input_test, batch_input_pred_test_AE))
            metrics.mean_relative_error_latent_encoder(
                    relative_error(batch_latent_test, batch_latent_pred_test))
            metrics.mean_relative_error_data_decoder(
                    relative_error(batch_input_test, batch_input_pred_test_decoder))

        @tf.function
        def dist_test_step(batch_input_test, batch_latent_test):
            return dist_strategy.experimental_run_v2(test_step, (batch_input_test, batch_latent_test))
###############################################################################
#                             Train Neural Network                            #
###############################################################################
    print('Beginning Training')
    for epoch in range(hyperp.num_epochs):
        print('================================')
        print('            Epoch %d            ' %(epoch))
        print('================================')
        print('Case: ' + file_paths.case_name + '\n' + 'NN: ' + file_paths.NN_name + '\n')
        print('GPUs: ' + options.dist_which_gpus + '\n')
        print('Optimizing %d batches of size %d:' %(num_batches_train, hyperp.batch_size))
        start_time_epoch = time.time()
        batch_counter = 0
        total_loss_train = 0
        for batch_input_train, batch_latent_train in dist_input_and_latent_train:
            start_time_batch = time.time()
            #=== Computing Training Step ===#
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
        print('Train Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Decoder: %.3e, Aug: %.3e'\
                %(metrics.mean_loss_train, metrics.mean_loss_train_autoencoder.result(),
                    metrics.mean_loss_train_encoder.result(),
                    metrics.mean_loss_train_decoder.result(),
                    metrics.mean_loss_train_forward_model.result()))
        print('Val AE: %.3e, Encoder: %.3e, Decoder: %.3e'\
                %(metrics.mean_loss_val_autoencoder.result(),
                    metrics.mean_loss_val_encoder.result(),
                    metrics.mean_loss_val_decoder.result()))
        print('Test AE: %.3e, Encoder: %.3e, Decoder: %.3e'\
                %(metrics.mean_loss_test_autoencoder.result(),
                    metrics.mean_loss_test_encoder.result(),
                    metrics.mean_loss_test_decoder.result()))
        print('Rel Errors: AE: %.3e, Encoder: %.3e, Decoder: %.3e\n'\
                %(metrics.mean_relative_error_data_autoencoder.result(),
                    metrics.mean_relative_error_latent_encoder.result(),
                    metrics.mean_relative_error_data_decoder.result()))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Save Current Model and Metrics ===#
        if epoch % 5 == 0:
            NN.save_weights(file_paths.NN_savefile_name)
            metrics.save_metrics(file_paths)
            dump_attrdict_as_yaml(hyperp, file_paths.NN_savefile_directory, 'hyperp')
            dump_attrdict_as_yaml(options, file_paths.NN_savefile_directory, 'options')
            print('Current Model and Metrics Saved')

    #=== Save final model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    metrics.save_metrics(file_paths)
    dump_attrdict_as_yaml(hyperp, file_paths.NN_savefile_directory, 'hyperp')
    dump_attrdict_as_yaml(options, file_paths.NN_savefile_directory, 'options')
    print('Final Model Saved')
