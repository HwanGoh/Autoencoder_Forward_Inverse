#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""

import tensorflow as tf
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_autoencoder(data_pred, data_true):
    return tf.norm(tf.subtract(data_true, data_pred), 2, axis = 1)

def loss_encoder(latent_pred, latent_true, penalty):
    return penalty*tf.norm(tf.subtract(latent_pred, latent_true), 2, axis = 1)

def reg_prior(parameter, prior_mean, L_pr, penalty):
    if penalty != 0:
        return penalty*tf.norm(tf.linalg.matmul(tf.subtract(parameter, prior_mean), L_pr), 2, axis = 1)
    else:
        return 0

def KL_divergence(post_mean, post_var, cov_prior_inv, det_cov_prior):
    det_cov_post = tf.math.reduce_prod(post_var)
    trace_cov_prior_inv_times_cov_post = 
    return
    

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(prediction, true):
    return tf.norm(true - prediction, 2, axis = 1)/tf.norm(true, 2, axis = 1)