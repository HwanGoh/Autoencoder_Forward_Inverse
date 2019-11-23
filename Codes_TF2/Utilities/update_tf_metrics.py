#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:59:48 2019

@author: hwan
"""
import tensorflow as tf


###############################################################################
#                          Update Tensorflow Metrics                          #
###############################################################################
@tf.function
def update_tf_metrics_validation(NN, parameter_val, state_obs_val, loss_autoencoder, loss_forward_problem):
    parameter_pred_val_batch_AE = NN(parameter_val)
    state_pred_val_batch = NN.encoder(parameter_val)
    loss_val_batch_autoencoder = loss_autoencoder(parameter_pred_val_batch_AE, parameter_val)
    loss_val_batch_forward_problem = loss_forward_problem(state_pred_val_batch, state_obs_val, penalty)
    loss_val_batch = loss_val_batch_autoencoder + loss_val_batch_forward_problem
    return loss_val_batch, loss_val_batch_autoencoder, loss_val_batch_forward_problem

@tf.function
def update_tf_metrics_test(NN, penalty, parameter_test, state_obs_test, loss_autoencoder, loss_forward_problem, relative_error):
    parameter_pred_test_batch_AE = NN(parameter_test)
    parameter_pred_test_batch_Inverse_problem = NN.decoder(state_obs_test)
    state_pred_test_batch = NN.encoder(parameter_test)
    loss_test_batch_autoencoder = loss_autoencoder(parameter_pred_test_batch_AE, parameter_test)
    loss_test_batch_forward_problem = loss_forward_problem(state_pred_test_batch, state_obs_test, penalty)
    loss_test_batch = loss_test_batch_autoencoder + loss_test_batch_forward_problem
    relative_error_batch_parameter_autoencoder = relative_error(parameter_pred_test_batch_AE, parameter_test)
    relative_error_batch_parameter_inverse_problem = relative_error(parameter_pred_test_batch_Inverse_problem, parameter_test)
    relative_error_batch_state_obs = relative_error(state_pred_test_batch, state_obs_test)
    return loss_test_batch, loss_test_batch_autoencoder, loss_test_batch_forward_problem, relative_error_batch_parameter_autoencoder, relative_error_batch_parameter_inverse_problem, relative_error_batch_state_obs

@tf.function
def update_tf_metrics_tensorboard(NN, epoch, gradients, 
                                  loss_train_batch_average, loss_train_batch_average_autoencoder, loss_train_batch_average_forward_problem,
                                  loss_val_batch_average, loss_val_batch_average_autoencoder, loss_val_batch_average_forward_problem,
                                  loss_test_batch_average, loss_test_batch_average_autoencoder, loss_test_batch_average_forward_problem,
                                  relative_error_batch_average_parameter_autoencoder, relative_error_batch_average_parameter_inverse_problem, relative_error_batch_average_state_obs):
    tf.summary.scalar('loss_training', loss_train_batch_average.result(), step=epoch)
    tf.summary.scalar('loss_training_autoencoder', loss_train_batch_average_autoencoder.result(), step=epoch)
    tf.summary.scalar('loss_training_forward_problem', loss_train_batch_average_forward_problem.result(), step=epoch)
    tf.summary.scalar('loss_val', loss_val_batch_average.result(), step=epoch)
    tf.summary.scalar('loss_val_autoencoder', loss_val_batch_average_autoencoder.result(), step=epoch)
    tf.summary.scalar('loss_val_forward_problem', loss_val_batch_average_forward_problem.result(), step=epoch)
    tf.summary.scalar('loss_test', loss_test_batch_average.result(), step=epoch)
    tf.summary.scalar('loss_test_autoencoder', loss_test_batch_average_autoencoder.result(), step=epoch)
    tf.summary.scalar('loss_test_forward_problem', loss_test_batch_average_forward_problem.result(), step=epoch)
    tf.summary.scalar('relative_error_parameter_autoencoder', relative_error_batch_average_parameter_autoencoder.result(), step=epoch)
    tf.summary.scalar('relative_error_parameter_inverse_problem', relative_error_batch_average_parameter_inverse_problem.result(), step=epoch)
    tf.summary.scalar('relative_error_state_obs', relative_error_batch_average_state_obs.result(), step=epoch)
    for w in NN.weights:
        tf.summary.histogram(w.name, w, step=epoch)
    l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
    for gradient, variable in zip(gradients, NN.trainable_variables):
        tf.summary.histogram("gradients_norm/" + variable.name, l2_norm(gradient), step = epoch)