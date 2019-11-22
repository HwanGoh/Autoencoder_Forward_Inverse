#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:37:28 2019

@author: hwan
"""
import os
import sys
import time

import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Utilities.optimize_autoencoder import optimize

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class HyperParameters: # Set defaults, hyperparameters of interest will be overwritten later
    data_type         = 'full'
    num_hidden_layers = 7
    truncation_layer  = 4 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty           = 50
    batch_size        = 1000
    num_epochs        = 2
    gpu               = '0'

class RunOptions:
    def __init__(self, hyper_p):
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_training_data = 50000
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Random Seed ===#
        self.random_seed = 1234

###############################################################################
#                                 File Name                                   #
###############################################################################                
        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        
        #=== File name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if self.fin_dimensions_2D == 1:
            fin_dimension = ''
        if self.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, hyper_p.activation, penalty_string, self.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
############################################################################### 
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyper_p.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(self.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(self.num_training_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type

        #=== Saving Neural Network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        
        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":  
    ############################
    #   Hyperparameter Space   #
    ############################        
    num_hidden_layers = Integer(1, 6, name='num_hidden_layers')
    num_hidden_nodes = Integer(500, 2000, name='num_hidden_nodes')
    activation = Categorical(['elu', 'relu', 'tanh'], name='activation')
    penalty = Integer(1, 200, name='penalty')
    batch_size = Integer(100, 1000, name='batch_size')
    
    space = [num_hidden_layers, num_hidden_nodes, activation, penalty, batch_size]
    
    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(activation, num_hidden_layers, num_hidden_nodes, penalty, batch_size):
        #=== Instantiate Hyperparameters ===#
        hyper_p = HyperParameters()
        
        #=== GPU Settings ===#
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = hyper_p.gpu
        
        #=== Assign Hyperparameters of Interest ===#
        hyper_p.num_hidden_layers = num_hidden_layers
        hyper_p.truncation_layer = int(np.ceil(num_hidden_layers/2))
        hyper_p.num_hidden_nodes = num_hidden_nodes
        hyper_p.activation = activation
        hyper_p.penalty = penalty
        hyper_p.batch_size = batch_size
        
        #=== Instantiate Run_Options ===#
        run_options = RunOptions(hyper_p)
        
        #=== Loading Data ===#        
        obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val = load_thermal_fin_data(run_options, run_options.num_training_data, hyper_p.batch_size, run_options.random_seed) 
    
        #=== Neural Network ===#
        NN = AutoencoderFwdInv(hyper_p, run_options, parameter_dimension, run_options.full_domain_dimensions, obs_indices, run_options.NN_savefile_name)
        
        #=== Training ===#
        _, _, _, storage_array_loss_val, _, _, _, _, _, _, _, _  = optimize(hyper_p, run_options, NN, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, parameter_dimension, num_batches_train)
        
        return storage_array_loss_val[-1]
    
    ####################################
    #   Optimize for Hyperparameters   #
    ####################################
    #=== Minimize ===#
    res_gp = gp_minimize(objective_functional, space, n_calls=10, random_state=None)

    #=== Display Summary ===#
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(res_gp.fun))
    print('Optimized Parameters:')
    print('num_hidden_layers: {}'.format(res_gp.x[0]))
    print('num_hidden_nodes: {}'.format(res_gp.x[1]))
    print('activation: {}'.format(res_gp.x[2]))
    print('penalty: {}'.format(res_gp.x[3]))
    print('batch_size: {}'.format(res_gp.x[4]))
    
    #=== Save Outputs Name and Directory ===#    
    save_path = '../Hyperparameter_Optimization'
    name_of_file = 'best_set_of_hyperparameters.txt'
    complete_path_and_file = os.path.join(save_path, name_of_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #=== Write Hyperparameter Optimization Summary ===#
    with open(complete_path_and_file, 'w') as summary_txt:
        summary_txt.write('Optimized Validation Loss: {}\n'.format(res_gp.fun))
        summary_txt.write('\n')
        summary_txt.write('Optimized parameters:\n')
        summary_txt.write('num_hidden_layers: {}\n'.format(res_gp.x[0]))
        summary_txt.write('num_hidden_nodes: {}\n'.format(res_gp.x[1]))
        summary_txt.write('activation: {}\n'.format(res_gp.x[2]))
        summary_txt.write('penalty: {}\n'.format(res_gp.x[3]))
        summary_txt.write('batch_size: {}\n'.format(res_gp.x[4]))
            
    plot_convergence(res_gp)
    plt.savefig(save_path + '/' + str(int(time.time())) + '.png')
    
    
    
    





























