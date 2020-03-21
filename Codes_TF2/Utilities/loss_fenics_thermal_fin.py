#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""
import sys
sys.path.append('..')

import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_fenics(hyperp, run_options, V, solver, obs_indices, fenics_forward, state_obs_true, parameter_pred, penalty_aug):                 
    fenics_state_pred = fenics_forward(tf.math.exp(parameter_pred))
    fenics_state_pred = tf.cast(fenics_state_pred, dtype=tf.float32)
    return penalty_aug*tf.norm(tf.subtract(state_obs_true, fenics_state_pred, 2), axis = 1)



















