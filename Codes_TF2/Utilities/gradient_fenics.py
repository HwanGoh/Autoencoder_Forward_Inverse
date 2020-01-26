#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:41:25 2020

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
#                                   Gradients                                #
###############################################################################
def compute_gradient_fenics(hyperp, run_options, V, solver, obs_indices, state_obs_true, parameter_pred, penalty_aug, B_obs):    
    gradients = np.zeros((len(parameter_pred), V.dim()))
    for m in range(len(parameter_pred)):
        if run_options.data_thermal_fin_nine == 1:
            parameter_pred_dl = parameter_convert_nine(run_options, V, solver, parameter_pred[m,:])   
        if run_options.data_thermal_fin_vary == 1:
            parameter_pred_dl = convert_array_to_dolfin_function(V, parameter_pred[m,:])
        gradient = solver.gradient(parameter_pred_dl, state_obs_true[m,:].numpy(), B_obs)
        gradients[m,:] = gradient
    
    return gradients
        
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