#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:59:59 2019

@author: hwan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dl

from Generate_Data.forward_solve import Fin
from Generate_Data.thermal_fin import get_space
from Generate_Data.Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os

###############################################################################
#                               Parameters                                    #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 7
    truncation_layer  = 4 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 1
    num_training_data = 20
    batch_size        = 20
    num_epochs        = 20
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyper_p):
        #=== Random Seed ===#
        self.random_seed = 1234
        
        #=== Data type ===#
        self.use_full_domain_data = 0
        self.use_bnd_data = 0
        self.use_bnd_data_only = 0
        if hyper_p.data_type == 'full':
            self.use_full_domain_data = 1
        if hyper_p.data_type == 'bnd':
            self.use_bnd_data = 1
        if hyper_p.data_type == 'bndonly':
            self.use_bnd_data_only = 1
        
        #=== Observation Dimensions ===#
        self.full_domain_dimensions = 1446 
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        #=== Number of Testing Data ===#
        self.num_testing_data = 20
        
        #=== File name ===#
        if hyper_p.penalty >= 1:
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Loading and saving data ===#       
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test'
        self.figures_savefile_name_parameter_pred = self.figures_savefile_directory + '/' + 'parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_directory + '/' + 'state_pred'
        
        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Set hyperparameters ===#
    hyper_p = HyperParameters()
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    ##############################################
    #   Form Fenics Domain and Load Predictions  #
    ##############################################
    V,_ = get_space(40)
    solver = Fin(V) 
    
    df_parameter_test = pd.read_csv(run_options.NN_savefile_name + '_parameter_test' + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_parameter_pred = pd.read_csv(run_options.NN_savefile_name + '_parameter_pred' + '.csv')
    parameter_pred = df_parameter_pred.to_numpy()
    
    if run_options.use_full_domain_data == 1: # No state prediction if the truncation layer only consists of the observations
        df_state_test = pd.read_csv(run_options.NN_savefile_name + '_state_test' + '.csv')
        state_test = df_state_test.to_numpy()
        df_state_pred = pd.read_csv(run_options.NN_savefile_name + '_state_pred' + '.csv')
        state_pred = df_state_pred.to_numpy()
    
    ##############
    #  Plotting  #
    ##############
    #=== Plotting test parameter and test state ===#
    parameter_test_dl = solver.nine_param_to_function(parameter_test)
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
    parameter_pred_dl = solver.nine_param_to_function(parameter_pred)
    
    p_pred_fig = dl.plot(parameter_pred_dl)
    p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=18)  
    plt.savefig(run_options.figures_savefile_name_parameter_pred, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_parameter_pred) 
    plt.show()
    parameter_pred_error = np.linalg.norm(parameter_pred - parameter_test,2)/np.linalg.norm(parameter_test,2)
    print(parameter_pred_error)
    
    if run_options.use_full_domain_data == 1: # No state prediction if the truncation layer only consists of the observations
        state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
        s_pred_fig = dl.plot(state_pred_dl)
        s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=18)  
        plt.savefig(run_options.figures_savefile_name_state_pred, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_state_pred) 
        plt.show()
        state_pred_error = np.linalg.norm(state_pred - state_test,2)/np.linalg.norm(state_test,2)
        print(state_pred_error)