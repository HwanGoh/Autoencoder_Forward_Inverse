#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:12:38 2019

@author: hwan

To avoid using dolfin when training the neural network, the data generation and file is separated. Run this to generate and save thermal fin data.
"""

import numpy as np
import pandas as pd
import dolfin as dl
from gaussian_field import make_cov_chol
from forward_solve import Fin
from thermal_fin import get_space
import os

def generate_thermal_fin_data(num_data, generate_nine_parameters, generate_full_domain):
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following Sheroze's "test_thermal_fin_gradient.py" code
    
    V = get_space(40)
    solver = Fin(V) 
    
    print(V.dim())
    
    # Create storage arrays
    if generate_nine_parameters == 1:
        parameter = np.zeros((num_data, 9))
    if generate_full_domain == 1:
        parameter = np.zeros((num_data, V.dim()))
    state = np.zeros((num_data, V.dim()))
    
    
    for m in range(num_data):
        print('\nGenerating Parameters and Data Set %d of %d' %(m+1, num_data))
        # Generate parameters
        if generate_nine_parameters == 1:
            parameter[m,:], parameter_dl = parameter_generator_nine_values(V,solver)
        if generate_full_domain == 1:
            parameter[m,:], parameter_dl = parameter_generator_full_domain(V,solver)              
        # Solve PDE for state variable
        state_dl, _ = solver.forward(parameter_dl)
        state[m,:] = state_dl.vector().get_local()   
        
    return parameter, state

def parameter_generator_nine_values(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    generated_parameter = solver.subfin_avg_op(parameter_dl)
    parameter_dl = solver.nine_param_to_function(generated_parameter)
    
    return generated_parameter, parameter_dl

def parameter_generator_full_domain(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    parameter_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_dl))
    generated_parameter = parameter_dl.vector().get_local()
    
    return generated_parameter, parameter_dl

def convert_array_to_dolfin_function(V, nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    
    return nodal_values_dl  

###############################################################################
#                                  Executor                                   #
###############################################################################
if __name__ == "__main__":  
    
    # Select parameter type
    generate_nine_parameters = 1
    generate_full_domain = 0
    
    # Select true or test set
    generate_true_data = 0
    generate_test_data = 1
    
    # Defining filenames and creating directories
    num_training_data = 1
    parameter_true_savefilepath = '../Data/' + 'parameter_true_%d' %(num_training_data) 
    state_true_savefilepath = '../Data/' + 'state_true_%d' %(num_training_data) 
    parameter_test_savefilepath = '../Data/' + 'parameter_test'
    state_test_savefilepath = '../Data/' + 'state_test'
    if not os.path.exists('../Data'):
            os.makedirs('../Data')
    
    # Generating data
    if generate_true_data == 1:
        parameter_savefilepath = parameter_true_savefilepath
        state_savefilepath = state_true_savefilepath
        
    if generate_test_data == 1:
        parameter_savefilepath = parameter_test_savefilepath
        state_savefilepath = state_test_savefilepath
        
    parameter_data, state_data = generate_thermal_fin_data(num_training_data, generate_nine_parameters, generate_full_domain)
    df_parameter_data = pd.DataFrame({'parameter_data': parameter_data.flatten()})
    df_state_data = pd.DataFrame({'state_data': state_data.flatten()})
    df_parameter_data.to_csv(parameter_savefilepath + '.csv', index=False)  
    df_state_data.to_csv(state_savefilepath + '.csv', index=False)  
    print('\nData Saved')

    
