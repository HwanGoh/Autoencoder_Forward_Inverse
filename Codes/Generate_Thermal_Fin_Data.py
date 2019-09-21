#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:12:38 2019

@author: hwan
"""

import numpy as np
import dolfin as dl
from gaussian_field import make_cov_chol
from forward_solve import Fin
from thermal_fin import get_space

def generate_thermal_fin_data(num_data):
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following Sheroze's "test_thermal_fin_gradient.py" code
    V = get_space(40)
    solver = Fin(V) 
    
    print(V.dim())
    
    parameter = np.zeros((num_data, V.dim()))
    state = np.zeros((num_data, V.dim()))
    
    for m in range(num_data):
        print('\nGenerating Parameters and Data Set %d of %d' %(m+1, num_data))
        # Randomly generate piecewise constant true parameter with 9 values
        parameter[m,:], parameter_dl = parameter_generator_nine_values(V,solver) # True conductivity values               
        # Solve PDE for state variable
        state_dl, _ = solver.forward(parameter_dl)
        state[m,:] = state_dl.vector().get_local()   
        
    return parameter, state

def parameter_generator_nine_values(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_true_dl = convert_array_to_dolfin_function(V,generated_parameter)
    parameter_true_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_true_dl))
    generated_parameter = parameter_true_dl.vector().get_local()
    
    return generated_parameter, parameter_true_dl

def convert_array_to_dolfin_function(V, nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    
    return nodal_values_dl     
