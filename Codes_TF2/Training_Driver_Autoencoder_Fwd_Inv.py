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
    num_hidden_layers = 3
    truncation_layer  = 2 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 200
    penalty           = 1
    num_training_data = 50000
    batch_size        = 1000
    num_epochs        = 2000
    gpu               = '2'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #===  Number of Testing Data ===#
        self.num_testing_data = 200
        
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
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
        
        #=== Observation Dimensions === #
        self.full_domain_dimensions = 1446 
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        #=== File name ===#
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Loading and saving data ===#
        if self.use_full_domain_data == 1:
            self.observation_indices_savefilepath = '../Datasets/' + 'thermal_fin_full_domain'
            self.parameter_train_savefilepath = '../Datasets/' + 'parameter_train_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../Datasets/' + 'state_train_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Datasets/' + 'parameter_test_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Datasets/' + 'state_test_%d' %(self.num_testing_data) 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.observation_indices_savefilepath = '../Datasets/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../Datasets/' + 'parameter_train_bnd_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../Datasets/' + 'state_train_bnd_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Datasets/' + 'parameter_test_bnd_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Datasets/' + 'state_test_bnd_%d' %(self.num_testing_data)             
        
        #=== Saving neural network ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files

        #=== Creating Directories ===#
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyper_p, run_options):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = hyper_p.gpu
    
    #=== Loading Data ===#        
    obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val = load_thermal_fin_data(run_options, hyper_p.num_training_data, hyper_p.batch_size, run_options.random_seed) 
    
    #=== Neural Network ===#
    NN = AutoencoderFwdInv(hyper_p, run_options, parameter_dimension, run_options.full_domain_dimensions, obs_indices, run_options.NN_savefile_name, construct_flag = 1)
    
    #=== Training ===#
    storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem, storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem, storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs = optimize(hyper_p, run_options, NN, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, parameter_dimension, num_batches_train)

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
        hyper_p.num_training_data = int(sys.argv[6])
        hyper_p.batch_size        = int(sys.argv[7])
        hyper_p.num_epochs        = int(sys.argv[8])
        hyper_p.gpu               = str(sys.argv[9])
            
    #=== Set run options ===#         
    run_options = RunOptions(hyper_p)
    
    #=== Initiate training ===#
    trainer(hyper_p, run_options) 
    
     
     
     
     
     
     
     
     
     
     
     
     