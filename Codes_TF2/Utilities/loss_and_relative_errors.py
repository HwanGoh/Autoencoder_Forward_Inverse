#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""

import tensorflow as tf

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_autoencoder(parameter_pred, parameter_true):
    return tf.norm(tf.subtract(parameter_true, parameter_pred), 2, axis = 1)

def loss_encoder(latent_pred, latent_true, penalty):
    return penalty*tf.norm(tf.subtract(latent_pred, latent_true), 2, axis = 1)

def reg_prior(parameter, prior_mean, L_pr, penalty):
    return penalty*tf.norm(tf.linalg.matmul(L_pr, tf.subtract(parameter, prior_mean)), 2, axis = 1)

###############################################################################
#                               Relative Error                                #
###############################################################################
def relative_error(prediction, true):
    return tf.norm(true - prediction, 2, axis = 1)/tf.norm(true, 2, axis = 1)