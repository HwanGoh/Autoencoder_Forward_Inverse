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

def reg_prior(parameter, prior_mean, L_pr, penalty):
    if penalty != 0:
        return penalty*tf.math.square(tf.norm(tf.linalg.matmul(
            tf.subtract(parameter, prior_mean), L_pr), 2, axis = 1))
    else:
        return 0

def loss_forward_model(hyperp, run_options, V, solver, obs_indices, forward_model,
        state_obs_true, parameter_pred, penalty_aug):
    forward_model_state_pred = forward_model(parameter_pred)
    forward_model_state_pred = tf.cast(forward_model_state_pred, dtype=tf.float32)
    return penalty_aug*tf.math.square(tf.norm(tf.subtract(state_obs_true,
        forward_model_state_pred, 2), axis = 1))

def KLD_diagonal_post_cov(post_mean, log_post_var,
        prior_mean, prior_cov_inv,
        log_det_prior_cov, latent_dimension):
    trace_prior_cov_inv_times_cov_post = tf.reduce_sum(
            tf.multiply(tf.linalg.diag_part(prior_cov_inv), tf.math.exp(log_post_var)),
            axis=1)
    pdb.set_trace()
    prior_weighted_prior_mean_minus_post_mean = tf.reduce_sum(
            tf.multiply(tf.transpose(prior_mean - post_mean),
                tf.linalg.matmul(prior_cov_inv, tf.transpose(prior_mean - post_mean))),
            axis = 0)
    log_det_prior_cov_divide_det_cov_post =\
            log_det_prior_cov - tf.math.reduce_sum(log_post_var, axis=1)
    return 0.5*(trace_prior_cov_inv_times_cov_post +
            prior_weighted_prior_mean_minus_post_mean -
            latent_dimension + log_det_prior_cov_divide_det_cov_post)

def KLD_full_post_cov(post_mean, post_var, prior_mean,
        prior_cov_inv, det_prior_cov, latent_dimension):
    return 0

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(true, pred):
    return tf.keras.losses.mean_absolute_percentage_error(true, pred)
