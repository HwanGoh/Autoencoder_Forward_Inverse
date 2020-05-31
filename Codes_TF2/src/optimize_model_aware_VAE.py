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

from metrics_model_aware_VAE import Metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths,
        NN, loss_autoencoder, KLD_loss, relative_error, prior_cov,
        data_and_latent_train, data_and_latent_val, data_and_latent_test,
        data_dimension, latent_dimension, num_batches_train):

    #=== Matrix Determinants and Inverse of Prior Covariance ===#
    prior_cov_inv = np.linalg.inv(prior_cov)
    (sign, logdet) = np.linalg.slogdet(prior_cov)
    log_det_prior_cov = sign*logdet

    #=== Optimizer ===#
    optimizer = tf.keras.optimizers.Adam()

    #=== Define Metrics ===#
    metrics = Metrics()

    #=== Creating Directory for Trained Neural Network ===#
    if not os.path.exists(file_paths.NN_savefile_directory):
        os.makedirs(file_paths.NN_savefile_directory)

    #=== Tensorboard ===# "tensorboard --logdir=Tensorboard"
    if os.path.exists(file_paths.tensorboard_directory):
        shutil.rmtree(file_paths.tensorboard_directory)
    summary_writer = tf.summary.create_file_writer(file_paths.tensorboard_directory)

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    #=== Train Step ===#
    @tf.function
    def train_step(batch_data_train, batch_latent_train):
        with tf.GradientTape() as tape:
            batch_likelihood_train = NN(batch_data_train)
            batch_post_mean_train, batch_log_post_var_train = NN.encoder(batch_data_train)
            batch_loss_train_VAE = loss_autoencoder(batch_likelihood_train, batch_data_train)
            batch_loss_train_KLD = KLD_loss(batch_post_mean_train, batch_log_post_var_train,
                    tf.zeros(latent_dimension), prior_cov_inv, log_det_prior_cov, latent_dimension)
            batch_loss_train = -(batch_loss_train_VAE - batch_loss_train_KLD)
        gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
        optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
        metrics.mean_loss_train(batch_loss_train)
        metrics.mean_loss_train_autoencoder(batch_loss_train_VAE)
        metrics.mean_loss_train_encoder(batch_loss_train_KLD)
        return gradients

    #=== Validation Step ===#
    @tf.function
    def val_step(batch_data_val, batch_latent_val):
        batch_likelihood_val = NN(batch_data_val)
        batch_post_mean_val, batch_log_post_var_val = NN.encoder(batch_data_val)
        batch_loss_val_VAE = loss_autoencoder(batch_likelihood_val, batch_data_val)
        batch_loss_val_KLD = KLD_loss(batch_post_mean_val, batch_log_post_var_val,
                tf.zeros(latent_dimension), prior_cov_inv, log_det_prior_cov, latent_dimension)
        batch_loss_val = -(batch_loss_val_VAE - batch_loss_val_KLD)
        metrics.mean_loss_val_autoencoder(batch_loss_val_VAE)
        metrics.mean_loss_val_encoder(batch_loss_val_KLD)
        metrics.mean_loss_val(batch_loss_val)

    #=== Test Step ===#
    @tf.function
    def test_step(batch_data_test, batch_latent_test):
        batch_likelihood_test = NN(batch_data_test)
        batch_post_mean_test, batch_log_post_var_test = NN.encoder(batch_data_test)
        batch_data_pred_test = NN.decoder(batch_latent_test)
        batch_loss_test_VAE = loss_autoencoder(batch_likelihood_test, batch_data_test)
        batch_loss_test_KLD = KLD_loss(batch_post_mean_test, batch_log_post_var_test,
                tf.zeros(latent_dimension), prior_cov_inv, log_det_prior_cov, latent_dimension)
        batch_loss_test = -(batch_loss_test_VAE - batch_loss_test_KLD)
        metrics.mean_loss_test_autoencoder(batch_loss_test_VAE)
        metrics.mean_loss_test_encoder(batch_loss_test_KLD)
        metrics.mean_loss_test(batch_loss_test)

        metrics.mean_relative_error_data_autoencoder(relative_error(batch_likelihood_test, batch_data_test))
        metrics.mean_relative_error_latent_encoder(relative_error(tf.math.exp(batch_post_mean_test),
            batch_latent_test))
        metrics.mean_relative_error_data_decoder(relative_error(batch_data_pred_test, batch_data_test))

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
        if epoch == 0:
            initial_sum_gradient_norms = 0
        initial_sum_gradient_norms, relative_gradient_norm =\
                metrics.update_tensorboard(summary_writer, epoch, NN,
                        gradients, initial_sum_gradient_norms)

        #=== Update Storage Arrays ===#
        metrics.update_storage_arrays(relative_gradient_norm)

        #=== Display Epoch Iteration Information ===#
        elapsed_time_epoch = time.time() - start_time_epoch
        print('Time per Epoch: %.4f\n' %(elapsed_time_epoch))
        print('Train Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_train.result(),
                    metrics.mean_loss_train_autoencoder.result(),
                    metrics.mean_loss_train_encoder.result()))
        print('Val Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_val.result(),
                    metrics.mean_loss_val_autoencoder.result()
                    , metrics.mean_loss_val_encoder.result()))
        print('Test Loss: Full: %.3e, AE: %.3e, Encoder: %.3e'\
                %(metrics.mean_loss_test.result(),
                    metrics.mean_loss_test_autoencoder.result(),
                    metrics.mean_loss_test_encoder.result()))
        print('Rel Errors: AE: %.3e, Encoder: %.3e, Decoder: %.3e\n'\
                %(metrics.mean_relative_error_data_autoencoder.result(),
                    metrics.mean_relative_error_latent_encoder.result(),
                    metrics.mean_relative_error_data_decoder.result()))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.update_storage_arrays(relative_gradient_norm)

        #=== Saving Metrics ===#
        if epoch %100 ==0:
            NN.save_weights(file_paths.NN_savefile_name)
            metrics.save_metrics(file_paths)
            print('Current Model and Metrics Saved')

    #=== Save Final Model ===#
    NN.save_weights(file_paths.NN_savefile_name)
    metrics.save_metrics(file_paths)
    print('Final Model and Metrics Saved')
