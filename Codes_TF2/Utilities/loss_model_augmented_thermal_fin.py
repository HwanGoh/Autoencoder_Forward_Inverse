#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""
import sys
sys.path.append('..')

import tensorflow as tf
import dolfin as dl
import numpy as np

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_model_augmented(hyperp, run_options, V, solver, obs_indices, state_obs_true, autoencoder_pred, penalty_aug):
    if run_options.data_thermal_fin_nine == 1:
        autoencoder_pred_dl = parameter_convert_nine(run_options, V, solver, autoencoder_pred)   
    if run_options.data_thermal_fin_vary == 1:
        autoencoder_pred_dl = convert_array_to_dolfin_function(V, autoencoder_pred)
    state_dl, _ = solver.forward(autoencoder_pred_dl)    
    state_data = state_dl.vector().get_local()
    if hyperp.data_type == 'bnd':
        state_data = state_data[obs_indices]    
    
    return penalty_aug*tf.norm(tf.subtract(state_obs_true, state_data), 2, axis = 1)

###############################################################################
#                              Fenics Functions                               #
###############################################################################
def parameter_convert_nine(run_options, V, solver, autoencoder_pred):
    parameter_dl = solver.nine_param_to_function(autoencoder_pred)
    if run_options.fin_dimensions_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
        parameter_values = parameter_dl.vector().get_local()  
        zero_indices = np.where(parameter_values == 0)[0]
        for ind in zero_indices:
            parameter_values[ind] = parameter_values[ind-1]
        parameter_dl = convert_array_to_dolfin_function(V, parameter_values) 
    return parameter_dl

def convert_array_to_dolfin_function(V, nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    return nodal_values_dl  


















