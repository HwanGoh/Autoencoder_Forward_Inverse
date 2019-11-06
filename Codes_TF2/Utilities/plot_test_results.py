#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:59:59 2019

@author: hwan
"""

import numpy as np
import dolfin as dl
from forward_solve import Fin
from thermal_fin import get_space
from Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function
import matplotlib.pyplot as plt

def plot_test_results(run_options, parameter_test, parameter_pred, state_pred):
    #####################################
    #   Form Test Parameters and State  #
    #####################################
    V,_ = get_space(40)
    solver = Fin(V) 
    
    ##############
    #  Plotting  #
    ##############
    #=== Plotting test parameter and test state ===#
    parameter_test_dl = solver.nine_param_to_function(parameter_test.T)
    state_test_dl, _ = solver.forward(parameter_test_dl) # generate true state for comparison
    state_test = state_test_dl.vector().get_local()         
    
    p_test_fig = dl.plot(parameter_test_dl)
    p_test_fig.ax.set_title('True Parameter', fontsize=18)  
    plt.savefig(run_options.figures_savefile_name_parameter_test, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_parameter_test)   
    plt.show()
    
    s_test_fig = dl.plot(state_test_dl)
    s_test_fig.ax.set_title('True State', fontsize=18) 
    plt.savefig(run_options.figures_savefile_name_state_test, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_state_test) 
    plt.show()
    
    #=== Plotting predictions of test parameter and test state ===#
    parameter_pred_dl = solver.nine_param_to_function(parameter_pred.T)
    
    p_pred_fig = dl.plot(parameter_pred_dl)
    p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=18)  
    plt.savefig(run_options.figures_savefile_name_parameter_pred, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_parameter_pred) 
    plt.show()
    parameter_pred_error = np.linalg.norm(parameter_pred - parameter_test,2)/np.linalg.norm(parameter_test,2)
    print(parameter_pred_error)
    
    if run_options.use_full_domain_data == 1 or run_options.use_bnd_data == 1: # No state prediction if the truncation layer only consists of the observations
        state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
        s_pred_fig = dl.plot(state_pred_dl)
        s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=18)  
        plt.savefig(run_options.figures_savefile_name_state_pred, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_state_pred) 
        plt.show()
        state_pred_error = np.linalg.norm(state_pred - state_test,2)/np.linalg.norm(state_test,2)
        print(state_pred_error)