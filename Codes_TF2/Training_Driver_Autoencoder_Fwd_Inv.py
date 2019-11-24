#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""
import os
import sys

import tensorflow as tf
import pandas as pd

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
from Utilities.NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Utilities.loss_and_relative_errors import loss_autoencoder, loss_forward_problem, relative_error
from Utilities.optimize_autoencoder import optimize
from Utilities.optimize_distributed_autoencoder import optimize_distributed

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    num_hidden_layers = 7
    truncation_layer  = 4 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty           = 50
    batch_size        = 100
    num_epochs        = 20
    
class RunOptions:
    def __init__(self): 
        #=== Use Distributed Strategy ===#
        self.use_distributed_training = 1
        
        #=== Which GPUs to use ===#
        self.dist_which_gpus = '1,2,3'
        
        #=== Use Single GPU ===#
        self.which_gpu = '1'
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_training_data = 50000
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if self.data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if self.data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions

###############################################################################
#                                 File Paths                                  #
###############################################################################  
class FilePaths():              
    def __init__(self, hyperp, run_options): 
        #=== Declaring File Name Components ===#
        if run_options.data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if run_options.data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if run_options.fin_dimensions_2D == 1:
            fin_dimension = ''
        if run_options.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyperp.penalty >= 1:
            hyperp.penalty = int(hyperp.penalty)
            penalty_string = str(hyperp.penalty)
        else:
            penalty_string = str(hyperp.penalty)
            penalty_string = 'pt' + penalty_string[2:]
        
        #=== File Name ===#
        self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_string, run_options.num_training_data, hyperp.batch_size, hyperp.num_epochs)

        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyperp.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(run_options.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(run_options.num_training_data) + fin_dimension + '_' + hyperp.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(run_options.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(run_options.num_testing_data) + fin_dimension + '_' + hyperp.data_type + parameter_type
        
        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.tensorboard_directory = '../Tensorboard/' + self.filename

###############################################################################
#                                  Training                                   #
###############################################################################
def trainer(hyperp, run_options, file_paths):
    #=== GPU Settings ===# Must put this first! Because TF2 will automatically work on a GPU and it may clash with used ones if the visible device list is not yet specified
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        pdb.set_trace()
        
    #=== Load Data ===#       
    obs_indices, parameter_train, state_obs_train,\
    parameter_test, state_obs_test,\
    data_input_shape, parameter_dimension\
    = load_thermal_fin_data(file_paths, run_options.num_training_data, run_options.num_testing_data, run_options.parameter_dimensions)    
    
    #=== Construct Validation Set and Batches ===#   
    parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val,\
    num_training_data, num_batches_train, num_batches_val\
    = form_train_val_test_batches(run_options.num_training_data, parameter_train, state_obs_train, parameter_test, state_obs_test, hyperp.batch_size, run_options.random_seed)
    
    #=== Non-distributed Training ===#
    if run_options.use_distributed_training == 0:        
        #=== Neural Network ===#
        NN = AutoencoderFwdInv(hyperp, parameter_dimension, run_options.full_domain_dimensions, obs_indices)
        
        #=== Training ===#
        storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem,\
        storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem,\
        storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem,\
        storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs\
        = optimize(hyperp, run_options, file_paths, NN, loss_autoencoder, loss_forward_problem, relative_error,\
                   parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val,\
                   parameter_dimension, num_batches_train)
    
    #=== Distributed Training ===#
    if run_options.use_distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            NN = AutoencoderFwdInv(hyperp, parameter_dimension, run_options.full_domain_dimensions, obs_indices)
            
            #=== Training ===#
            storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem,\
            storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem,\
            storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem,\
            storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs\
            = optimize_distributed(dist_strategy, hyperp, run_options, file_paths, NN, loss_autoencoder, loss_forward_problem, relative_error,\
                                   parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val,\
                                   parameter_dimension, num_batches_train)

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
    df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics" + '.csv', index=False)

###############################################################################
#                                 Driver                                      #
###############################################################################     
if __name__ == "__main__":     

    #=== Hyperparameters and Run Options ===#    
    hyperp = Hyperparameters()
    run_options = RunOptions()
    
    if len(sys.argv) > 1:
        hyperp.data_type         = str(sys.argv[1])
        hyperp.num_hidden_layers = int(sys.argv[2])
        hyperp.truncation_layer  = int(sys.argv[3])
        hyperp.num_hidden_nodes  = int(sys.argv[4])
        hyperp.penalty           = float(sys.argv[5])
        hyperp.batch_size        = int(sys.argv[6])
        hyperp.num_epochs        = int(sys.argv[7])
        run_options.which_gpu    = str(sys.argv[8])

    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)

    #=== Initiate training ===#
    trainer(hyperp, run_options, file_paths) 
    
     
     
     
     
     
     
     
     
     
     
     
     