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
def loss_autoencoder(data_pred, data_true):
    return tf.norm(tf.subtract(data_true, data_pred), 2, axis = 1)

def loss_encoder_or_decoder(pred, true, penalty):
    return penalty*tf.norm(tf.subtract(pred, true), 2, axis = 1)

def reg_prior(parameter, prior_mean, L_pr, penalty):
    if penalty != 0:
        return penalty*tf.norm(tf.linalg.matmul(tf.subtract(parameter, prior_mean), L_pr), 2, axis = 1)
    else:
        return 0

def KLD_diagonal_post_cov(post_mean, post_var, prior_mean, prior_cov_inv, det_prior_cov, latent_dimension):    
    trace_prior_cov_inv_times_cov_post = tf.reduce_sum(tf.multiply(tf.linalg.diag_part(prior_cov_inv), post_var), axis=1)
    prior_weighted_prior_mean_minus_post_mean = tf.linalg.diag_part(tf.linalg.matmul(prior_mean - post_mean, tf.linalg.matmul(prior_cov_inv, tf.transpose(prior_mean - post_mean))))              
    log_det_prior_cov_divide_det_cov_post = tf.math.log(det_prior_cov) # Going to form determinant by adding log(1/var) one by one to avoid numerical error
    for n in range(0, post_var.shape[1]):
        log_det_prior_cov_divide_det_cov_post += tf.math.log(1/post_var[:,n])
    return 0.5*(trace_prior_cov_inv_times_cov_post + prior_weighted_prior_mean_minus_post_mean - latent_dimension + log_det_prior_cov_divide_det_cov_post)

def KLD_full_post_cov(post_mean, post_var, prior_mean, prior_cov_inv, det_prior_cov, latent_dimension): 
    return 0

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(prediction, true):
    return tf.norm(true - prediction, 2, axis = 1)/tf.norm(true, 2, axis = 1)