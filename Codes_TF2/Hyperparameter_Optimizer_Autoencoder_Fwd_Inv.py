#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:37:28 2019

@author: hwan
"""
import os
import sys
import shutil # for deleting directories

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt import dump, load

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.form_train_val_test_batches import form_train_val_test_batches
from Utilities.NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Utilities.loss_and_relative_errors import loss_autoencoder, loss_forward_problem, relative_error
from Utilities.optimize_autoencoder import optimize
from Utilities.optimize_distributed_autoencoder import optimize_distributed

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                      Hyperparameters and Run_Options                        #
###############################################################################
class Hyperparameters: # Choose defaults, hyperparameters of interest wil be overwritten later
    data_type         = 'full'
    num_hidden_layers = 6
    truncation_layer  = 4 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 519
    activation        = 'relu'
    penalty           = 1
    batch_size        = 1000
    num_epochs        = 1000
    
class RunOptions:
    def __init__(self): 
        #=== Use Distributed Strategy ===#
        self.use_distributed_training = 1
        
        #=== Which GPUs to Use for Distributed Strategy ===#
        self.dist_which_gpus = '0,1,2'
        
        #=== Which Single GPU to Use ===#
        self.which_gpu = '1'
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_data_train = 50000
        self.num_data_test = 200
        
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
        self.filename = self.dataset + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
 
        #=== Loading and Saving Data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyperp.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(run_options.num_data_train) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(run_options.num_data_train) + fin_dimension + '_' + hyperp.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(run_options.num_data_test) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(run_options.num_data_test) + fin_dimension + '_' + hyperp.data_type + parameter_type

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.hyperp_opt_Trained_NNs_directory = '../Hyperparameter_Optimization/Trained_NNs'
        self.hyperp_opt_Tensorboard_directory = '../Hyperparameter_Optimization/Tensorboard'
        self.NN_savefile_directory = self.hyperp_opt_Trained_NNs_directory + '/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.tensorboard_directory = self.hyperp_opt_Tensorboard_directory + '/' + self.filename

        #=== Saving Hyperparameter Optimization Outputs  ===#    
        self.hyperp_opt_outputs_directory = '../Hyperparameter_Optimization'
        self.hyperp_opt_skopt_res_savefile_name = self.hyperp_opt_outputs_directory + '/hyperp_opt_result.pkl'
        self.hyperp_opt_optimal_parameters_savefile_name = self.hyperp_opt_outputs_directory + '/optimal_set_of_hyperparameters.txt'
        self.hyperp_opt_scenarios_trained_savefile_name = self.hyperp_opt_outputs_directory + '/scenarios_trained.txt'
        self.hyperp_opt_validation_losses_savefile_name = self.hyperp_opt_outputs_directory + '/validation_losses.csv'
        self.hyperp_opt_convergence_savefile_name = self.hyperp_opt_outputs_directory + '/convergence.png'

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":  
    ###################################
    #   Select Optimization Options   #
    ###################################
    #=== Number of Iterations ===#
    n_calls = 25
    
    #=== Select Hyperparameters of Interest ===# Note: you can just manually create a space of variables instead of using a dictionary, but I prefer to have the list of variable names on hand for use in the outputs later as well as the tuple to act as an argument to the objective function
    hyperp_of_interest_dict = {}
    #hyperp_of_interest_dict['num_hidden_layers'] = Integer(5, 20, name='num_hidden_layers')
    #hyperp_of_interest_dict['num_hidden_nodes'] = Integer(500, 2000, name='num_hidden_nodes')
    #hyperp_of_interest_dict['penalty'] = Integer(1, 200, name='penalty')
    hyperp_of_interest_dict['activation'] = Categorical(['elu', 'relu', 'tanh'], name='activation')
    hyperp_of_interest_dict['batch_size'] = Integer(20, 1000, name='batch_size')
    
    #####################
    #   Initial Setup   #
    #####################
    hyperp_of_interest_list = list(hyperp_of_interest_dict.keys()) 
    hyperp_of_interest_objective_args_tuple = tuple(hyperp_of_interest_list)
    
    #=== Generate skopt 'space' list ===#
    space = []
    for key, val in hyperp_of_interest_dict.items(): 
        space.append(val)
    
    #=== Instantiate Hyperparameters and Run Options to Load Data ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()
    file_paths = FilePaths(hyperp, run_options)
    
    #=== GPU Settings ===# Must put this first! Because TF2 will automatically work on a GPU and it may clash with used ones if the visible device list is not yet specified
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
    
    #=== Load Data ===#       
    obs_indices, parameter_train, state_obs_train,\
    parameter_test, state_obs_test,\
    data_input_shape, parameter_dimension\
    = load_thermal_fin_data(file_paths, run_options.num_data_train, run_options.num_data_test, run_options.parameter_dimensions)    
    
    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_objective_args_tuple):            
        #=== Assign Hyperparameters of Interest ===#
        for key, val in hyperp_of_interest_objective_args_tuple.items(): 
            setattr(hyperp, key, val)        
        hyperp.truncation_layer = int(np.ceil(hyperp.num_hidden_layers/2))
        
        #=== Update File Paths with New Hyperparameters ===#
        file_paths = FilePaths(hyperp, run_options)
        
        #=== Construct Validation Set and Batches ===# 
        if run_options.use_distributed_training == 0:
            GLOBAL_BATCH_SIZE = hyperp.batch_size
        if run_options.use_distributed_training == 1:
            GLOBAL_BATCH_SIZE = hyperp.batch_size * len(gpus) # To avoid the core dump issue, have to do this instead of hyperp.batch_size * dist_strategy.num_replicas_in_sync
        parameter_and_state_obs_train, parameter_and_state_obs_val, parameter_and_state_obs_test,\
        run_options.num_data_train, num_data_val, run_options.num_data_test,\
        num_batches_train, num_batches_val, num_batches_test\
        = form_train_val_test_batches(parameter_train, state_obs_train, parameter_test, state_obs_test, GLOBAL_BATCH_SIZE, run_options.random_seed)
    
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
                       parameter_and_state_obs_train, parameter_and_state_obs_val, parameter_and_state_obs_test,\
                       parameter_dimension, num_batches_train)
        
        #=== Distributed Training ===#
        if run_options.use_distributed_training == 1:
            dist_strategy = tf.distribute.MirroredStrategy()
            GLOBAL_BATCH_SIZE = hyperp.batch_size*dist_strategy.num_replicas_in_sync
            with dist_strategy.scope():
                #=== Neural Network ===#
                NN = AutoencoderFwdInv(hyperp, parameter_dimension, run_options.full_domain_dimensions, obs_indices)
                
            #=== Training ===#
            storage_array_loss_train, storage_array_loss_train_autoencoder, storage_array_loss_train_forward_problem,\
            storage_array_loss_val, storage_array_loss_val_autoencoder, storage_array_loss_val_forward_problem,\
            storage_array_loss_test, storage_array_loss_test_autoencoder, storage_array_loss_test_forward_problem,\
            storage_array_relative_error_parameter_autoencoder, storage_array_relative_error_parameter_inverse_problem, storage_array_relative_error_state_obs\
            = optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE, hyperp, run_options, file_paths, NN, loss_autoencoder, loss_forward_problem, relative_error,\
                                   parameter_and_state_obs_train, parameter_and_state_obs_val, parameter_and_state_obs_test,\
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
        
        return storage_array_loss_val[-1]
    
    ################################
    #   Optimize Hyperparameters   #
    ################################
    hyperp_opt_result = gp_minimize(objective_functional, space, n_calls=n_calls, random_state=None)

    ##################################
    #   Display Optimal Parameters   #
    ##################################
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
    print('Optimized Parameters:')    
    for n, parameter_name in enumerate(hyperp_of_interest_list):
        print(parameter_name + ': {}'.format(hyperp_opt_result.x[n]))
    
    #####################################
    #   Save Optimization Information   #
    #####################################
    #=== Save .pkl File ===#
    dump(hyperp_opt_result, file_paths.hyperp_opt_skopt_res_savefile_name, store_objective=False)
    
    #=== Write Optimal Set Hyperparameters ===#
    with open(file_paths.hyperp_opt_optimal_parameters_savefile_name, 'w') as optimal_set_txt:
        optimal_set_txt.write('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
        optimal_set_txt.write('\n')
        optimal_set_txt.write('Optimized parameters:\n')      
        for n, parameter_name in enumerate(hyperp_of_interest_list):
            optimal_set_txt.write(parameter_name + ': {}\n'.format(hyperp_opt_result.x[n]))
            
    #=== Write List of Scenarios Trained ===#
    with open(file_paths.hyperp_opt_scenarios_trained_savefile_name, 'w') as scenarios_trained_txt:
        for scenario in hyperp_opt_result.x_iters:
            scenarios_trained_txt.write("%s\n" % scenario)      
            
    #=== Write List of Validation Losses ===#
    validation_losses_dict = {}
    validation_losses_dict['validation_losses'] = hyperp_opt_result.func_vals
    df_validation_losses = pd.DataFrame(validation_losses_dict)
    df_validation_losses.to_csv(file_paths.hyperp_opt_validation_losses_savefile_name, index=False)    
        
    #=== Convergence Plot ===#    
    plot_convergence(hyperp_opt_result)
    plt.savefig(file_paths.hyperp_opt_convergence_savefile_name)
    
    print('Outputs Saved')
    
    #####################################################
    #   Delete All Suboptimal Trained Neural Networks   #
    #####################################################
    #=== Assigning hyperp with Optimal Hyperparameters ===#
    for num, parameter in enumerate(hyperp_of_interest_list): 
        setattr(hyperp, parameter, hyperp_opt_result.x[num])
    hyperp.truncation_layer = int(np.ceil(hyperp.num_hidden_layers/2))
    
    #=== Updating File Paths with Optimal Hyperparameters ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Deleting Suboptimal Neural Networks ===#
    directories_list_Trained_NNs = os.listdir(path=file_paths.hyperp_opt_Trained_NNs_directory)
    directories_list_Tensorboard = os.listdir(path=file_paths.hyperp_opt_Tensorboard_directory)
    for filename in directories_list_Trained_NNs:
        if filename != file_paths.filename:
            shutil.rmtree(file_paths.hyperp_opt_Trained_NNs_directory + '/' + filename) 
            shutil.rmtree(file_paths.hyperp_opt_Tensorboard_directory + '/' + filename) 
    
    print('Suboptimal Trained Networks Deleted')
    





























