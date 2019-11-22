#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:37:28 2019

@author: hwan
"""
import os
import sys
import time
import shutil # for deleting directories

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
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
        self.num_training_data = 200
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 1
        self.fin_dimensions_3D = 0
        
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

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.hyper_p_opt_Trained_NNs_directory = '../Hyperparameter_Optimization/Trained_NNs'
        self.hyper_p_opt_Tensorboard_directory = '../Hyperparameter_Optimization/Tensorboard'
        self.NN_savefile_directory = self.hyper_p_opt_Trained_NNs_directory + '/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.tensorboard_directory = self.hyper_p_opt_Tensorboard_directory + '/' + self.filename

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":  
    #####################
    #   Initial Setup   #
    #####################
    #=== Select Hyperparameters of Interest ===# Note: you can just manually create a space of variables instead of using a dictionary, but I prefere to have the list of variable names on hand
    hyper_p_of_interest_dict = {}
    hyper_p_of_interest_dict['num_hidden_layers'] = Integer(1, 20, name='num_hidden_layers')
    hyper_p_of_interest_dict['num_hidden_nodes'] = Integer(500, 2000, name='num_hidden_nodes')
    hyper_p_of_interest_dict['activation'] = Categorical(['elu', 'relu', 'tanh'], name='activation')
    hyper_p_of_interest_dict['penalty'] = Integer(1, 200, name='penalty')
    hyper_p_of_interest_dict['batch_size'] = Integer(100, 1000, name='batch_size')
    
    hyper_p_of_interest_list = list(hyper_p_of_interest_dict.keys()) 
    hyper_p_of_interest_objective_args_tuple = tuple(hyper_p_of_interest_list)

    #=== Generate skopt 'space' list ===#
    space = []
    for key, val in hyper_p_of_interest_dict.items(): 
        space.append(val)
    
    #=== Instantiate Hyperparameters and Run Options to Load Data ===#
    hyper_p = HyperParameters()
    run_options = RunOptions(hyper_p)
    
    #=== Load Data ===#
    obs_indices, parameter_train, state_obs_train, parameter_test, state_obs_test, data_input_shape, parameter_dimension = load_thermal_fin_data(run_options, run_options.num_training_data) 
    
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = hyper_p.gpu
    
    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyper_p_of_interest_objective_args_tuple):            
        #=== Assign Hyperparameters of Interest ===#
        for key, val in hyper_p_of_interest_objective_args_tuple.items(): 
            setattr(hyper_p, key, val)
        
        hyper_p.truncation_layer = int(np.ceil(hyper_p.num_hidden_layers/2))
        
        #=== Update Run_Options ===#
        run_options = RunOptions(hyper_p)
        
        #=== Construct Validation Set and Batches ===#        
        parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, num_training_data, num_batches_train, num_batches_val = form_train_val_test_batches(run_options.num_training_data, parameter_train, state_obs_train, parameter_test, state_obs_test, hyper_p.batch_size, run_options.random_seed)
    
        #=== Neural Network ===#
        NN = AutoencoderFwdInv(hyper_p, run_options, parameter_dimension, run_options.full_domain_dimensions, obs_indices, run_options.NN_savefile_name)
        
        #=== Training ===#
        storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs  = optimize(hyper_p, run_options, NN, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, parameter_dimension, num_batches_train)
    
        #=== Saving Metrics ===#
        metrics_dict = {}
        metrics_dict['loss_train'] = storage_array_loss_train
        metrics_dict['loss_train_autoencoder'] = storage_array_loss_train_autoencoder
        metrics_dict['loss_train_forward_problem'] = storage_array_loss_train_forward_problem
        metrics_dict['loss_val'] = storage_array_loss_val
        metrics_dict['loss_val_autoencoder'] = storage_array_loss_val_autoencoder
        metrics_dict['loss_val_forward_problem'] = storage_array_loss_val_forward_problem
        metrics_dict['relative_error_parameter_autoencoder'] = storage_array_relative_error_parameter_autoencoder
        metrics_dict['relative_error_parameter_inverse_problem'] = storage_array_relative_error_parameter_inverse_problem
        metrics_dict['relative_error_state_obs'] = storage_array_relative_error_state_obs
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(run_options.NN_savefile_name + "_metrics" + '.csv', index=False)     
        
        return storage_array_loss_val[-1]
    
    ####################################
    #   Optimize for Hyperparameters   #
    ####################################
    #=== Minimize ===#
    res_gp = gp_minimize(objective_functional, space, n_calls=10, random_state=None)

    ######################################
    #   Save Optimized Hyperparameters   #
    ######################################
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(res_gp.fun))
    print('Optimized Parameters:')
    
    for n, parameter_name in enumerate(hyper_p_of_interest_list):
        print(parameter_name + ': {}'.format(res_gp.x[n]))
    
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
        for n, parameter_name in enumerate(hyper_p_of_interest_list):
            summary_txt.write(parameter_name + ': {}\n'.format(res_gp.x[n]))
        
    #=== Convergence Plot ===#    
    plot_convergence(res_gp)
    plt.savefig(save_path + '/' + 'convergence.png')
    
    #=== Delete All Suboptimal Trained Neural Networks ===#
    hyper_p.num_hidden_layers = res_gp.x[0]
    hyper_p.truncation_layer = int(np.ceil(hyper_p.num_hidden_layers/2))
    hyper_p.num_hidden_nodes = res_gp.x[1]
    hyper_p.activation = res_gp.x[2]
    hyper_p.penalty = res_gp.x[3]
    hyper_p.batch_size = res_gp.x[4]
    
    run_options = RunOptions(hyper_p)
    
    directories_list_Trained_NNs = os.listdir(path=run_options.hyper_p_opt_Trained_NNs_directory)
    directories_list_Tensorboard = os.listdir(path=run_options.hyper_p_opt_Tensorboard_directory)
    for filename in directories_list_Trained_NNs:
        if filename != run_options.filename:
            shutil.rmtree(run_options.hyper_p_opt_Trained_NNs_directory + '/' + filename) 
            shutil.rmtree(run_options.hyper_p_opt_Tensorboard_directory + '/' + filename) 
    
    
    





























