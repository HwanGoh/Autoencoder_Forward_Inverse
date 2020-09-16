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
    return penalty*tf.keras.losses.mean_squared_error(true, pred)

def loss_weighted_penalized_difference(true, pred, weight_matrix, penalty):
    return penalty*tf.keras.losses.mean_squared_error(
            tf.linalg.matmul(true, tf.transpose(weight_matrix)),
            tf.linalg.matmul(pred, tf.transpose(weight_matrix)))

def reg_prior(parameter, prior_mean, prior_covariance_cholesky_inverse, penalty):
    if penalty != 0:
        return penalty*tf.math.square(tf.norm(
            tf.linalg.matmul(
                tf.subtract(parameter, prior_mean),
                tf.transpose(prior_covariance_cholesky_inverse)), 2, axis = 1))
    else:
        return 0

def loss_forward_model(hyperp, options,
                       forward_model,
                       state_obs_true, parameter_pred,
                       penalty):
    forward_model_state_pred = forward_model(parameter_pred)
    forward_model_state_pred = tf.cast(forward_model_state_pred, dtype=tf.float32)
    return penalty*tf.keras.losses.mean_squared_error(state_obs_true,
            forward_model_state_pred)

def KLD_diagonal_post_cov(post_mean, log_post_var,
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
    return penalty*0.5*(trace_prior_cov_inv_times_cov_post +
            prior_weighted_prior_mean_minus_post_mean -
            latent_dimension + log_det_prior_cov_divide_det_cov_post)

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(true, pred):
    return tf.keras.losses.mean_absolute_percentage_error(true, pred)
