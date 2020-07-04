#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:53:31 2019

@author: hwan
"""
import sys
sys.path.append('../../../..')

import shutil # for deleting directories
import os
import time

import tensorflow as tf
import dolfin as dl
dl.set_log_level(30)
import numpy as np
import pandas as pd

from metrics_model_induced_autoencoder import Metrics

from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                             Training Properties                             #
###############################################################################
def optimize(hyperp, run_options, file_paths,
        NN, optimizer,
        obs_indices,
        loss_penalized_difference, loss_forward_model,
        relative_error, reg_prior, L_pr,
        input_and_latent_train, input_and_latent_val, input_and_latent_test,
        input_dimensions,
        num_batches_train):

    #=== Generate Dolfin Function Space and Mesh ===# These are in the scope and used below in the
                                                    # Fenics forward function and gradient
    if run_options.fin_dimensions_2D == 1:
        V, mesh = get_space_2D(40)
    if run_options.fin_dimensions_3D == 1:
        V, mesh = get_space_3D(40)
    solver = Fin(V)
    parameter_pred_dl = dl.Function(V)
    print(V.dim())
    if run_options.data_thermal_fin_nine == 1:
        B_obs = solver.observation_operator()
    else:
        B_obs = np.zeros((len(obs_indices), V.dim()), dtype=np.float32)
        B_obs[np.arange(len(obs_indices)), obs_indices.flatten()] = 1

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
#                     Fenics Forward Functions and Gradient                   #
###############################################################################
    @tf.custom_gradient
    def fenics_forward(parameter_pred):
        fenics_state_pred = np.zeros((parameter_pred.shape[0], len(obs_indices)))
        # for m in range(parameter_pred.shape[0]):
        #     parameter_pred_dl.vector().set_local(parameter_pred[m,:].numpy())
        #     state_dl, _ = solver.forward(parameter_pred_dl)
        #     state_data_values = state_dl.vector().get_local()
        #     if hyperp.data_type == 'full':
        #         fenics_state_pred[m,:] = state_data_values
        #     if hyperp.data_type == 'bnd':
        #         fenics_state_pred[m,:] = state_data_values[obs_indices].flatten()
        def fenics_forward_grad(dy):
            fenics_forward_grad = np.zeros((parameter_pred.shape[0], parameter_pred.shape[1]))
            # for m in range(parameter_pred.shape[0]):
            #     Jac_forward = solver.sensitivity(parameter_pred_dl, B_obs)
            #     fenics_forward_grad[m,:] = tf.linalg.matmul(tf.expand_dims(dy[m,:],0), Jac_forward)
            return fenics_forward_grad
        return fenics_state_pred, fenics_forward_grad

###############################################################################
#                   Training, Validation and Testing Step                     #
###############################################################################
    #=== Train Step ===# NOTE: NOT YET CODED FOR REVERSE AUTOENCODER. Becareful of the logs and exp
    #@tf.function
    def train_step(batch_input_train, batch_latent_train):
        with tf.GradientTape() as tape:
            if run_options.use_standard_autoencoder == 1:
                batch_input_pred_train_AE = NN(tf.math.log(batch_input_train))
                batch_latent_pred_train = NN.encoder(tf.math.log(batch_input_train))
                batch_input_pred_train = NN.decoder(batch_latent_train)
                batch_loss_train_autoencoder = loss_penalized_difference(
                        batch_input_pred_train_AE, tf.math.log(batch_input_train), 1)
                batch_loss_train_encoder = loss_penalized_difference(
                        batch_latent_pred_train, batch_latent_train, hyperp.penalty_encoder)
                batch_loss_train_decoder = loss_penalized_difference(
                        batch_input_pred_train, batch_input_train, hyperp.penalty_decoder)
                batch_loss_train_forward_model = loss_forward_model(
                        hyperp, run_options,
                        fenics_forward,
                        batch_latent_train, tf.math.exp(batch_input_pred_train_AE),
                        hyperp.penalty_aug)
            if run_options.use_reverse_autoencoder == 1:
                batch_state_obs_train = batch_input_train
                batch_parameter_pred = NN.encoder(batch_input_train)
            batch_loss_train = batch_loss_train_autoencoder + batch_loss_train_encoder +\
                    batch_loss_train_decoder + batch_loss_train_forward_model
        gradients = tape.gradient(batch_loss_train, NN.trainable_variables)
        optimizer.apply_gradients(zip(gradients, NN.trainable_variables))
        metrics.mean_loss_train(batch_loss_train)
        metrics.mean_loss_train_autoencoder(batch_loss_train_autoencoder)
        metrics.mean_loss_train_encoder(batch_loss_train_encoder)
        metrics.mean_loss_train_decoder(batch_loss_train_decoder)
        metrics.mean_loss_train_forward_model(batch_loss_train_forward_model)
        return gradients

    #=== Validation Step ===#
    #@tf.function
    def val_step(batch_input_val, batch_latent_val):
        if run_options.use_standard_autoencoder == 1:
            batch_input_pred_val_AE = NN(tf.math.log(batch_input_val))
            batch_latent_pred_val = NN.encoder(tf.math.log(batch_input_val))
            batch_input_pred_val = NN.decoder(batch_latent_val)
            batch_loss_val_autoencoder = loss_penalized_difference(
                    batch_input_pred_val_AE, tf.math.log(batch_input_val), 1)
            batch_loss_val_encoder = loss_penalized_difference(
                    batch_latent_pred_val, batch_latent_val, hyperp.penalty_encoder)
            batch_loss_val_decoder = loss_penalized_difference(
                    batch_input_pred_val, batch_input_val, hyperp.penalty_decoder)
            batch_loss_val_forward_model = loss_forward_model(
                    hyperp, run_options,
                    fenics_forward, batch_latent_val, tf.math.exp(batch_input_pred_val_AE),
                    hyperp.penalty_aug)
        if run_options.use_reverse_autoencoder == 1:
            batch_state_obs_val = batch_input_val
            batch_parameter_pred = NN.encoder(batch_input_val)
        batch_loss_val = batch_loss_val_autoencoder + batch_loss_val_encoder +\
                batch_loss_val_decoder + batch_loss_val_forward_model
        metrics.mean_loss_val_autoencoder(batch_loss_val_autoencoder)
        metrics.mean_loss_val_encoder(batch_loss_val_encoder)
        metrics.mean_loss_val_decoder(batch_loss_val_decoder)
        metrics.mean_loss_val_forward_model(batch_loss_val_forward_model)
        metrics.mean_loss_val(batch_loss_val)

    #=== Test Step ===#
    #@tf.function
    def test_step(batch_input_test, batch_latent_test):
        if run_options.use_standard_autoencoder == 1:
            batch_input_pred_test_AE = NN(tf.math.log(batch_input_test))
            batch_latent_pred_test = NN.encoder(tf.math.log(batch_input_test))
            batch_input_pred_test_decoder = NN.decoder(batch_latent_test)

            batch_loss_test_autoencoder = loss_penalized_difference(
                    batch_input_pred_test_AE, tf.math.log(batch_input_test), 1)
            batch_loss_test_encoder = loss_penalized_difference(
                    batch_latent_pred_test, batch_latent_test, hyperp.penalty_encoder)
            batch_loss_test_decoder = loss_penalized_difference(
                    batch_input_pred_test_decoder, batch_input_test, hyperp.penalty_decoder)
            batch_loss_test_forward_model = loss_forward_model(
                    hyperp, run_options,
                    fenics_forward, batch_latent_test, tf.math.exp(batch_input_pred_test_AE),
                    hyperp.penalty_aug)

            metrics.mean_relative_error_input_autoencoder(
                    relative_error(batch_input_pred_test_AE, batch_input_test))
            metrics.mean_relative_error_latent_encoder(
                    relative_error(batch_latent_pred_test, batch_latent_test))
            metrics.mean_relative_error_input_decoder(
                    relative_error(batch_input_pred_test_decoder, batch_input_test))
        if run_options.use_reverse_autoencoder == 1:
            batch_state_obs_test = batch_input_test
            batch_parameter_pred = NN.encoder(batch_input_test)
        batch_loss_test = batch_loss_test_autoencoder + batch_loss_test_encoder +\
                batch_loss_test_decoder + batch_loss_test_forward_model
        metrics.mean_loss_test_autoencoder(batch_loss_test_autoencoder)
        metrics.mean_loss_test_encoder(batch_loss_test_encoder)
        metrics.mean_loss_test_decoder(batch_loss_test_decoder)
        metrics.mean_loss_test_forward_model(batch_loss_test_forward_model)
        metrics.mean_loss_test(batch_loss_test)

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
        print('Train Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Decoder: %.3e, Aug: %.3e'\
                %(metrics.mean_loss_train.result(), metrics.mean_loss_train_autoencoder.result(),
                    metrics.mean_loss_train_encoder.result(),
                    metrics.mean_loss_train_decoder.result(),
                    metrics.mean_loss_train_forward_model.result()))
        print('Val Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Decoder: %.3e, Aug: %.3e'\
                %(metrics.mean_loss_val.result(), metrics.mean_loss_val_autoencoder.result(),
                    metrics.mean_loss_val_encoder.result(), metrics.mean_loss_val_decoder.result(),
                    metrics.mean_loss_val_forward_model.result()))
        print('Test Loss: Full: %.3e, AE: %.3e, Encoder: %.3e, Decoder: %.3e, Aug: %.3e'\
                %(metrics.mean_loss_test.result(), metrics.mean_loss_test_autoencoder.result(),
                    metrics.mean_loss_test_encoder.result(), metrics.mean_loss_test_decoder.result(),
                    metrics.mean_loss_test_forward_model.result()))
        print('Rel Errors: AE: %.3e, Encoder: %.3e, Decoder: %.3e'\
                %(metrics.mean_relative_error_input_autoencoder.result(),
                    metrics.mean_relative_error_latent_encoder.result(),
                    metrics.mean_relative_error_input_decoder.result()))
        print('Relative Gradient Norm: %.4f\n' %(metrics.relative_gradient_norm))
        start_time_epoch = time.time()

        #=== Resetting Metrics ===#
        metrics.reset_metrics()

        #=== Save Current Model ===#
        if epoch % 5 == 0:
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
