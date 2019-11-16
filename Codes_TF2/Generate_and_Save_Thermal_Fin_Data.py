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
import matplotlib as plt
from Generate_Thermal_Fin_Data.gaussian_field import make_cov_chol
from Generate_Thermal_Fin_Data.forward_solve import Fin
from Generate_Thermal_Fin_Data.thermal_fin import get_space
import os
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def generate_thermal_fin_data(num_data, generate_nine_parameters, generate_varying, generate_boundary_state):
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following Sheroze's "test_thermal_fin_gradient.py" code
    
    # Generate Dolfin function space and mesh
    V, mesh = get_space(40)
    solver = Fin(V)    
    
    print(V.dim())    
    
    # Create storage arrays
    if generate_nine_parameters == 1:
        parameter = np.zeros((num_data, 9))
    if generate_varying == 1:
        parameter = np.zeros((num_data, V.dim()))
    if generate_boundary_state == 1:
        obs_indices = list(set(sum((f.entities(0).tolist() for f in dl.SubsetIterator(solver.boundaries, 1)), []))) # entries of this vector represent which of the (V.dim() x 1) vector of domain indices correspond to the boundary; NOT the degrees of freedom  
        state = np.zeros((num_data, len(obs_indices)))
        # check if actually boundary points
        #mesh_coordinates = mesh.coordinates()
        #obs_coor = np.zeros((len(obs_indices),2))        
        #obs_counter = 0
        #for ind in obs_indices:
        #    obs_coor[obs_counter,:] = mesh_coordinates[ind,:]
        #    obs_counter = obs_counter + 1
        #dl.plot(mesh)
        
    else:
        state = np.zeros((num_data, V.dim()))
        obs_indices = list(range(V.dim()))
        
    for m in range(num_data):
        print('\nGenerating Parameters and Data Set %d of %d\n' %(m+1, num_data))
        # Generate parameters
        if generate_nine_parameters == 1:
            parameter[m,:], parameter_dl = parameter_generator_nine_values(V,solver)
        if generate_varying == 1:
            parameter[m,:], parameter_dl = parameter_generator_varying(V,solver)              
        # Solve PDE for state variable
        state_dl, _ = solver.forward(parameter_dl)        
        if generate_boundary_state == 1:
            state_full_domain = state_dl.vector().get_local()
            state[m,:] = state_full_domain[obs_indices]
        else:
            state[m,:] = state_dl.vector().get_local()             
        
    return parameter, state, obs_indices

def parameter_generator_nine_values(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    generated_parameter = solver.subfin_avg_op(parameter_dl)
    parameter_dl = solver.nine_param_to_function(generated_parameter)
    
    return generated_parameter, parameter_dl

def parameter_generator_varying(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    #parameter_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_dl))
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

    #=== Number of Data ===#
    num_data = 50000
    
    #===  Select parameter type ===#
    generate_nine_parameters = 0
    generate_varying = 1
        
    #=== Select true or test set ===#
    generate_train_data = 1
    generate_test_data = 0

    #===  Select observation type ===#
    generate_full_domain = 1
    generate_boundary_state = 0
    
    #===  Defining filenames and creating directories ===#         
    if generate_nine_parameters == 1:
        parameter_type = '_nine'
        
    if generate_varying == 1:
        parameter_type = '_vary'
        
    if generate_full_domain == 1:    
        observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_full_domain'
        parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(num_data) + parameter_type
        state_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(num_data) + parameter_type
        parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(num_data) + parameter_type
        state_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(num_data) + parameter_type  
        
    if generate_boundary_state == 1:
        observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_bnd_indices'
        parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_bnd_%d' %(num_data) + parameter_type
        state_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_bnd_%d' %(num_data) + parameter_type
        parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_bnd_%d' %(num_data) + parameter_type
        state_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_bnd_%d' %(num_data) + parameter_type         
 
    if generate_train_data == 1:
        parameter_savefilepath = parameter_train_savefilepath
        state_savefilepath = state_train_savefilepath
        
    if generate_test_data == 1:
        parameter_savefilepath = parameter_test_savefilepath
        state_savefilepath = state_test_savefilepath
    
    #=== Generating data ===#   
    parameter_data, state_data, obs_indices = generate_thermal_fin_data(num_data, generate_nine_parameters, generate_varying, generate_boundary_state)
    df_obs_indices = pd.DataFrame({'obs_indices': obs_indices})
    df_obs_indices.to_csv(observation_indices_savefilepath + '.csv', index=False)  
    df_parameter_data = pd.DataFrame({'parameter_data': parameter_data.flatten()})
    df_state_data = pd.DataFrame({'state_data': state_data.flatten()})
    df_parameter_data.to_csv(parameter_savefilepath + '.csv', index=False)  
    df_state_data.to_csv(state_savefilepath + '.csv', index=False)  
    print('\nData Saved to ' + parameter_savefilepath + ' and ' + state_savefilepath)

