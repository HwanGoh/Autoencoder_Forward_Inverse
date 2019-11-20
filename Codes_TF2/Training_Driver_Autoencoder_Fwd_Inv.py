#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""

import numpy as np
import pandas as pd

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Utilities.optimize_autoencoder import optimize

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys

np.random.seed(1234)

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 50
    batch_size        = 1000
    num_epochs        = 2000
    gpu               = '2'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #=== Data Set ===#
        data_thermal_fin_nine = 1
        data_thermal_fin_vary = 0
        
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

        self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, self.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
############################################################################### 
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyper_p.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(self.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(self.num_training_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        
        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files

        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                  Training                                   #
###############################################################################
def trainer(hyper_p, run_options):
    if run_options.fin_dimensions_2D == 1:
        #=== GPU Settings ===#
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = hyper_p.gpu
    
    #=== Loading Data ===#        
    obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val = load_thermal_fin_data(run_options, run_options.num_training_data, hyper_p.batch_size, run_options.random_seed) 
    
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

###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters ===#    
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.data_type         = str(sys.argv[1])
        hyper_p.num_hidden_layers = int(sys.argv[2])
        hyper_p.truncation_layer  = int(sys.argv[3])
        hyper_p.num_hidden_nodes  = int(sys.argv[4])
        hyper_p.penalty           = float(sys.argv[5])
        hyper_p.batch_size        = int(sys.argv[6])
        hyper_p.num_epochs        = int(sys.argv[7])
        hyper_p.gpu               = str(sys.argv[8])
        
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     