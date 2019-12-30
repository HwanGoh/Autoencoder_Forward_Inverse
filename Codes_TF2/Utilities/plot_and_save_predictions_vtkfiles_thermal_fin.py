#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:01:27 2019

@author: hwan
"""
from fenics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.append('../..')

from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function
from Utilities.save_paraview_png import save_paraview_png

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_and_save_predictions_vtkfiles(hyperp, run_options, file_paths):
###############################################################################
#                     Form Fenics Domain and Load Predictions                 #
###############################################################################
    #=== Form Fenics Domain ===#
    if run_options.fin_dimensions_2D == 1:
        V,_ = get_space_2D(40)
    if run_options.fin_dimensions_3D == 1:
        V, mesh = get_space_3D(40)
    
    solver = Fin(V) 
    
    #=== Load Observation Indices, Test and Predicted Parameters and State ===#
    df_obs_indices = pd.read_csv(file_paths.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 

    df_parameter_test = pd.read_csv(file_paths.savefile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_parameter_pred = pd.read_csv(file_paths.savefile_name_parameter_pred + '.csv')
    parameter_pred = df_parameter_pred.to_numpy()
    
    df_state_pred = pd.read_csv(file_paths.savefile_name_state_pred + '.csv')
    state_pred = df_state_pred.to_numpy()
    
###############################################################################
#                             Plotting Predictions                            #
###############################################################################
    #=== Converting Test Parameter Into Dolfin Object and Computing State Observation ===#       
    if run_options.data_thermal_fin_nine == 1:
        parameter_test_dl = solver.nine_param_to_function(parameter_test)
        if run_options.fin_dimensions_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
            parameter_values = parameter_test_dl.vector().get_local()  
            zero_indices = np.where(parameter_values == 0)[0]
            for ind in zero_indices:
                parameter_values[ind] = parameter_values[ind-1]
            parameter_test_dl = convert_array_to_dolfin_function(V, parameter_values)
    if run_options.data_thermal_fin_vary == 1:
        parameter_test_dl = convert_array_to_dolfin_function(V,parameter_test)
    
    state_test_dl, _ = solver.forward(parameter_test_dl) # generate true state for comparison
    state_test = state_test_dl.vector().get_local()    
    if hyperp.data_type == 'bnd':
        state_test = state_test[obs_indices].flatten()
        
    #=== Saving as vtkfile ===#
    vtkfile_parameter_test = File(file_paths.figures_savefile_name_parameter_test + '.pvd')
    vtkfile_parameter_test << parameter_test_dl
    vtkfile_state_test = File(file_paths.figures_savefile_name_state_test + '.pvd')
    vtkfile_state_test << state_test_dl
    
    #=== Converting Predicted Parameter into Dolfin Object ===# 
    if run_options.data_thermal_fin_nine == 1:
        parameter_pred_dl = solver.nine_param_to_function(parameter_pred)
        if run_options.fin_dimensions_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
            parameter_values = parameter_pred_dl.vector().get_local()  
            zero_indices = np.where(parameter_values == 0)[0]
            for ind in zero_indices:
                parameter_values[ind] = parameter_values[ind-1]
            parameter_pred_dl = convert_array_to_dolfin_function(V, parameter_values)
    if run_options.data_thermal_fin_vary == 1:
        parameter_pred_dl = convert_array_to_dolfin_function(V,parameter_pred)   
        
    if hyperp.data_type == 'full': # No visualization of state prediction if the truncation layer only consists of the boundary observations
        state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
    
    #=== Saving as vtkfile ===#
    vtkfile_parameter_pred = File(file_paths.figures_savefile_name_parameter_pred + '.pvd')
    vtkfile_parameter_pred << parameter_pred_dl
    vtkfile_state_pred = File(file_paths.figures_savefile_name_state_pred + '.pvd')
    vtkfile_state_pred << state_pred_dl
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    