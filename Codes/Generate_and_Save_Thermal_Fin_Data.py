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

###############################################################################
#                                  Executor                                   #
###############################################################################
if __name__ == "__main__":  
    
    num_training_data = 20
    data_savefilepath = '../Data/' + 'data_%d' %(num_training_data) 
    test_savefilepath = '../Data/' + 'test_%d' %(num_training_data) 
    if not os.path.exists('../Data'):
            os.makedirs('../Data')
    
    parameter_data, state_data = generate_thermal_fin_data(num_training_data)
    true_data = {'parameter_data': parameter_data.flatten(), 'state_data': state_data.flatten()}
    df = pd.DataFrame(true_data)   
    df.to_csv(data_savefilepath + '.csv', index=False)  
    print('\nData Saved')
    
    parameter_test, state_test = generate_thermal_fin_data(num_training_data)
    test_data = {'parameter_test': parameter_test.flatten(), 'state_test': state_test.flatten()}
    df = pd.DataFrame(test_data)   
    df.to_csv(test_savefilepath + '.csv', index=False)  
    print('\nData Saved')
    
