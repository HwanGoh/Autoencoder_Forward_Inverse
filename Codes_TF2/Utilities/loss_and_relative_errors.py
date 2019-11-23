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
@tf.function
def loss_autoencoder(autoencoder_pred, parameter_true):
    return tf.reduce_mean(tf.norm(tf.subtract(parameter_true, autoencoder_pred),2))

@tf.function
def loss_forward_problem(state_obs_pred, state_obs_true, penalty):
    return penalty*tf.reduce_mean(tf.norm(tf.subtract(state_obs_pred, state_obs_true),2))

###############################################################################
#                               Relative Error                                #
###############################################################################
@tf.function
def relative_error(prediction, true):
    return tf.norm(true - prediction, 2)/tf.norm(true, 2)