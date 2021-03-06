#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""
import numpy as np
import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_penalized_difference(true, pred, penalty):
    return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(true, pred)

def loss_weighted_penalized_difference(true, pred, weight_matrix, penalty):
    if weight_matrix.shape[0] == weight_matrix.shape[1]: # If matrix is square
        return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(
                tf.linalg.matmul(true, tf.transpose(weight_matrix)),
                tf.linalg.matmul(pred, tf.transpose(weight_matrix)))
    else: # Diagonal weight matrices with diagonals stored as rows
        return penalty*true.shape[1]*tf.keras.losses.mean_squared_error(
                tf.multiply(weight_matrix, true),
                tf.multiply(weight_matrix, pred))

def loss_forward_model(hyperp, options,
                       forward_model,
                       state_obs_true, parameter_pred,
                       penalty):
    forward_model_state_pred = forward_model(parameter_pred)
    forward_model_state_pred = tf.cast(forward_model_state_pred, dtype=tf.float32)
    return penalty*state_obs_true.shape[1]*tf.keras.losses.mean_squared_error(state_obs_true,
            forward_model_state_pred)

def loss_kld(post_mean, log_post_var,
             prior_mean, prior_cov_inv,
             log_det_prior_cov, latent_dimension,
             penalty):
    trace_prior_cov_inv_times_cov_post = tf.reduce_sum(
            tf.multiply(tf.linalg.diag_part(prior_cov_inv), tf.math.exp(log_post_var)),
            axis=1)
    prior_weighted_prior_mean_minus_post_mean = tf.reduce_sum(
            tf.multiply(tf.transpose(prior_mean - post_mean),
                tf.linalg.matmul(prior_cov_inv, tf.transpose(prior_mean - post_mean))),
            axis = 0)
    log_det_prior_cov_divide_det_cov_post =\
            log_det_prior_cov - tf.math.reduce_sum(log_post_var, axis=1)
    return penalty*(trace_prior_cov_inv_times_cov_post +
            prior_weighted_prior_mean_minus_post_mean -
            latent_dimension + log_det_prior_cov_divide_det_cov_post)

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(true, pred):
    return tf.keras.losses.mean_absolute_percentage_error(true, pred)
