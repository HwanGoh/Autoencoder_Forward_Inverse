#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:53:31 2019

@author: hwan
"""
import sys
sys.path.append('../..')

import shutil # for deleting directories
import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from metrics_VAE import Metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths,
             NN, optimizer,
             loss_penalized_difference, relative_error,
             prior_mean, prior_covariance_cholesky_inverse,
             input_and_latent_train, input_and_latent_val, input_and_latent_test,
             input_dimensions, latent_dimension,
             num_batches_train,
             loss_weighted_penalized_difference, noise_regularization_matrix,
             positivity_constraint):

    #=== Define Metrics ===#
    metrics = Metrics()

    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)

    #=== Tensorboard ===# "tensorboard --logdir=Tensorboard"
    if os.path.exists(file_paths.tensorboard_directory):
        shutil.rmtree(file_paths.tensorboard_directory)
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

    #=== Display Neural Network Architecture ===#
    NN.build((hyperp.batch_size, input_dimensions))
    NN.summary()

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    #=== Train Step ===#
    @tf.function
    def train_step(batch_input_train, batch_latent_train):
        with tf.GradientTape() as tape:
            batch_likelihood_train = NN(batch_input_train)
            batch_post_mean_train, batch_log_post_var_train = NN.encoder(batch_input_train)
            batch_posterior_sample_train = NN.IAF_chain_posterior((batch_post_mean_train,
                                                                   batch_log_post_var_train),
                                                                   sample_flag = True,
                                                                   infer_flag = False)

            batch_loss_train_VAE = loss_weighted_penalized_difference(
                    batch_input_train, batch_likelihood_train,
                    noise_regularization_matrix, 1)
            batch_loss_train_IAF_posterior = hyperp.penalty_IAF*\
                    NN.IAF_chain_posterior((batch_post_mean_train,
                                            batch_log_post_var_train),
                                            sample_flag = False,
                                            infer_flag = True)
            batch_loss_train_prior = loss_weighted_penalized_difference(
                    batch_latent_train, batch_posterior_sample_train,
                    prior_covariance_cholesky_inverse, hyperp.penalty_prior)

            batch_loss_train = -(-batch_loss_train_VAE\
                                 -batch_loss_train_IAF_posterior\
                                 -batch_loss_train_prior)

        gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
        optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
        metrics.mean_loss_train(batch_loss_train)
        metrics.mean_loss_train_VAE(batch_loss_train_VAE)
        metrics.mean_loss_train_encoder(batch_loss_train_IAF_posterior)
        metrics.mean_loss_train_prior(batch_loss_train_prior)

        return gradients

    #=== Validation Step ===#
    @tf.function
    def val_step(batch_input_val, batch_latent_val):
        batch_likelihood_val = NN(batch_input_val)
        batch_post_mean_val, batch_log_post_var_val = NN.encoder(batch_input_val)
        batch_posterior_sample_val = NN.IAF_chain_posterior((batch_post_mean_val,
                                                             batch_log_post_var_val),
                                                             sample_flag = True,
                                                             infer_flag = False)

        batch_loss_val_VAE = loss_weighted_penalized_difference(
                batch_input_val, batch_likelihood_val,
                noise_regularization_matrix, 1)
        batch_loss_val_IAF_posterior = hyperp.penalty_IAF*\
                NN.IAF_chain_posterior((batch_post_mean_val,
                                        batch_log_post_var_val),
                                        sample_flag = False,
                                        infer_flag = True)
        batch_loss_val_prior = loss_weighted_penalized_difference(
                batch_latent_val, batch_posterior_sample_val,
                prior_covariance_cholesky_inverse, hyperp.penalty_prior)

        batch_loss_val = -(-batch_loss_val_VAE\
                           -batch_loss_val_IAF_posterior\
                           -batch_loss_val_prior)

        metrics.mean_loss_val(batch_loss_val)
        metrics.mean_loss_val_VAE(batch_loss_val_VAE)
        metrics.mean_loss_val_encoder(batch_loss_val_IAF_posterior)
        metrics.mean_loss_val_prior(batch_loss_val_prior)

    #=== Test Step ===#
    @tf.function
    def test_step(batch_input_test, batch_latent_test):
        batch_likelihood_test = NN(batch_input_test)
        batch_post_mean_test, batch_log_post_var_test = NN.encoder(batch_input_test)
        batch_posterior_sample_test = NN.IAF_chain_posterior((batch_post_mean_test,
                                                              batch_log_post_var_test),
                                                              sample_flag = True,
                                                              infer_flag = False)
        batch_input_pred_test = NN.decoder(batch_latent_test)

        batch_loss_test_VAE = loss_weighted_penalized_difference(
                batch_input_test, batch_likelihood_test,
                noise_regularization_matrix, 1)
        batch_loss_test_IAF_posterior = hyperp.penalty_IAF*\
                NN.IAF_chain_posterior((batch_post_mean_test,
                                        batch_log_post_var_test),
                                        sample_flag = False,
                                        infer_flag = True)
        batch_loss_test_prior = loss_weighted_penalized_difference(
                batch_latent_test, batch_posterior_sample_test,
                prior_covariance_cholesky_inverse, hyperp.penalty_prior)

        batch_loss_test = -(-batch_loss_test_VAE\
                            -batch_loss_test_IAF_posterior\
                            -batch_loss_test_prior)

        metrics.mean_loss_test(batch_loss_test)
        metrics.mean_loss_test_VAE(batch_loss_test_VAE)
        metrics.mean_loss_test_encoder(batch_loss_test_IAF_posterior)
        metrics.mean_loss_test_prior(batch_loss_test_prior)

        metrics.mean_relative_error_input_VAE(relative_error(
            batch_input_test, batch_likelihood_test))
        metrics.mean_relative_error_latent_encoder(relative_error(
            batch_latent_test, batch_posterior_sample_test))
        metrics.mean_relative_error_input_decoder(relative_error(
            batch_input_test, batch_input_pred_test))

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
        for batch_num, (batch_input_train, batch_latent_train) in input_and_latent_train.enumerate():
            start_time_batch = time.time()
            #=== Computing Train Step ===#
            gradients = train_step(batch_input_train, batch_latent_train)
            elapsed_time_batch = time.time() - start_time_batch
            if batch_num  == 0:
                print('Time per Batch: %.4f' %(elapsed_time_batch))

        #=== Computing Relative Errors Validation ===#
        for batch_input_val, batch_latent_val in input_and_latent_val:
            val_step(batch_input_val, batch_latent_val)

        #=== Computing Relative Errors Test ===#
        for batch_input_test, batch_latent_test in input_and_latent_test:
            test_step(batch_input_test, batch_latent_test)

        #=== Update Current Relative Gradient Norm ===#
        with summary_writer.as_default():
            for w in NN.weights:
                tf.summary.histogram(w.name, w, step=epoch)
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            sum_gradient_norms = 0.0
            for gradient, variable in zip(gradients, NN.trainable_variables):
                tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient),
                        step = epoch)
                sum_gradient_norms += l2_norm(gradient)
                if epoch == 0:
                    initial_sum_gradient_norms = sum_gradient_norms
        metrics.relative_gradient_norm = sum_gradient_norms/initial_sum_gradient_norms

        #=== Track Training Metrics, Weights and Gradients ===#
        metrics.update_tensorboard(summary_writer, epoch)

        #=== Update Storage Arrays ===#
        metrics.update_storage_arrays()

        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, VAE: %.3e, IAF_post: %.3e, prior: %.3e'\
                %(metrics.mean_loss_train.result(),
                  metrics.mean_loss_train_VAE.result(),
                  metrics.mean_loss_train_encoder.result(),
                  metrics.mean_loss_train_prior.result()))
        print('Val Loss: Full: %.3e, VAE: %.3e, IAF_post: %.3e, prior: %.3e'\
                %(metrics.mean_loss_val.result(),
                  metrics.mean_loss_val_VAE.result(),
                  metrics.mean_loss_val_encoder.result(),
                  metrics.mean_loss_val_prior.result()))
        print('Test Loss: Full: %.3e, VAE: %.3e, IAF_post: %.3e, prior: %.3e'\
                %(metrics.mean_loss_test.result(),
                  metrics.mean_loss_test_VAE.result(),
                  metrics.mean_loss_test_encoder.result(),
                  metrics.mean_loss_test_prior.result()))
        print('Rel Errors: VAE: %.3e, Encoder: %.3e, Decoder: %.3e\n'\
                %(metrics.mean_relative_error_input_VAE.result(),
                  metrics.mean_relative_error_latent_encoder.result(),
                  metrics.mean_relative_error_input_decoder.result()))
        print('Relative Gradient Norm: %.4f\n' %(metrics.relative_gradient_norm))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Saving Current Model and Metrics ===#
        if epoch %100 == 0:
            NN.save_weights(file_paths.NN_savefile_name)
            metrics.save_metrics(file_paths)
            print('Current Model and Metrics Saved')

        #=== Gradient Norm Termination Condition ===#
        if metrics.relative_gradient_norm < 1e-6:
            print('Gradient norm tolerance reached, breaking training loop')
            break

    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    metrics.save_metrics(file_paths)
    print('Final Model and Metrics Saved')
