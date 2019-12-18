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
dl.set_log_level(30)
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_model_augmented(hyperp, run_options, V, solver, obs_indices, state_obs_true, parameter_pred, penalty_aug):
    fenics_state_pred = np.zeros((state_obs_true.shape[0], state_obs_true.shape[1]))
    for m in range(len(parameter_pred)):
        if run_options.data_thermal_fin_nine == 1:
            parameter_pred_dl = parameter_convert_nine(run_options, V, solver, parameter_pred[m,:])   
        if run_options.data_thermal_fin_vary == 1:
            parameter_pred_dl = convert_array_to_dolfin_function(V, parameter_pred[m,:])
        state_dl, _ = solver.forward(parameter_pred_dl)    
        state_data_values = state_dl.vector().get_local()
        if hyperp.data_type == 'full':
            fenics_state_pred[m,:] = state_data_values
        if hyperp.data_type == 'bnd':
            fenics_state_pred[m,:] = state_data_values[obs_indices].flatten()   
            
    return penalty_aug*tf.norm(tf.subtract(state_obs_true, fenics_state_pred), 2, axis = 1)

###############################################################################
#                              Fenics Functions                               #
###############################################################################
def parameter_convert_nine(run_options, V, solver, parameter_pred):
    parameter_dl = solver.nine_param_to_function(parameter_pred)
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


















