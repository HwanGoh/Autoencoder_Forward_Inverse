#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:51:00 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))
import shutil

import numpy as np
import pandas as pd

# Routine for outputting results
from hyperparameter_optimization_output import output_results

# Import FilePaths class and training routine
from Utilities.file_paths_VAE import FilePathsHyperparameterOptimization
from\
Utilities.hyperparameter_optimization_training_routine_custom_VAE_model_aware\
        import trainer_custom

# Import skopt code
from skopt.space import Real, Integer, Categorical

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                      Hyperparameters and Run_Options                        #
###############################################################################
class Hyperparameters:
    num_hidden_layers_encoder = 5
    num_hidden_layers_decoder = 2
    num_hidden_nodes_encoder  = 500
    num_hidden_nodes_decoder  = 500
    activation                = 'relu'
    penalty_KLD_incr          = 0.001
    penalty_KLD_rate          = 10
    penalty_post_mean         = 1
    num_data_train            = 500
    batch_size                = 100
    num_epochs                = 2

class RunOptions:
    #=== Use Distributed Strategy ===#
    distributed_training = 0

    #=== Which GPUs to Use for Distributed Strategy ===#
    dist_which_gpus = '0,1,2'

    #=== Which Single GPU to Use ===#
    which_gpu = '3'

    #=== Use Resnet ===#
    resnet = 0

    #=== Data Set Size ===#
    num_data_train_load = 5000
    num_data_test_load = 200
    num_data_test = 200

    #=== Data Properties ===#
    parameter_dimensions = 225
    obs_type = 'full'
    num_obs_points = 43

    #=== Noise Properties ===#
    add_noise = 0
    noise_level = 0.05
    num_noisy_obs = 20
    num_noisy_obs_unregularized = 20

    #=== Autocorrelation Prior Properties ===#
    prior_type_AC_train = 1
    prior_mean_AC_train = 2
    prior_variance_AC_train = 2.0
    prior_corr_AC_train = 0.5

    prior_type_AC_test = 1
    prior_mean_AC_test = 2
    prior_variance_AC_test = 2.0
    prior_corr_AC_test = 0.5

    #=== Matern Prior Properties ===#
    prior_type_matern_train = 0
    prior_kern_type_train = 'm32'
    prior_cov_length_train = 0.5

    prior_type_matern_test = 0
    prior_kern_type_test = 'm32'
    prior_cov_length_test = 0.5

    #=== Random Seed ===#
    random_seed = 4

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    ###################################
    #   Select Optimization Options   #
    ###################################
    #=== Number of Iterations ===#
    n_calls = 10

    #=== Select Hyperparameters of Interest ===#
    hyperp_of_interest_dict = {}
    hyperp_of_interest_dict['num_hidden_layers_encoder'] = Integer(5, 10,
            name='num_hidden_layers_encoder')
    hyperp_of_interest_dict['num_hidden_nodes_encoder'] = Integer(100, 1000,
            name='num_hidden_nodes_encoder')
    # hyperp_of_interest_dict['activation'] = Categorical(['relu', 'elu', 'sigmoid', 'tanh'], name='activation')
    hyperp_of_interest_dict['penalty_KLD_incr'] = Real(10, 1000, name='penalty_KLD_incr')
    hyperp_of_interest_dict['penalty_KLD_rate'] = Real(0, 1, name='penalty_KLD_rate')
    hyperp_of_interest_dict['penalty_post_mean'] = Real(10, 1000, name='penalty_post_mean')
    #hyperp_of_interest_dict['batch_size'] = Integer(100, 500, name='batch_size')

    #####################
    #   Initial Setup   #
    #####################
    #=== Generate skopt 'space' list ===#
    space = []
    for key, val in hyperp_of_interest_dict.items():
        space.append(val)

    #=== Instantiate Hyperparameters and Run Options to Load Data ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()
    run_options.model_aware = 1
    run_options.model_augmented = 0
    run_options.posterior_diagonal_covariance = 1
    run_options.posterior_IAF = 0
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(run_options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(run_options.parameter_dimensions)
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                                                     project_name,
                                                     data_options, dataset_directory)

    ################
    #   Training   #
    ################
    hyperp_opt_result = trainer_custom(hyperp, run_options, file_paths,
                                       n_calls, space,
                                       project_name,
                                       data_options, dataset_directory)

    ######################
    #   Output Results   #
    ######################
    output_results(hyperp, run_options, file_paths,
                   hyperp_of_interest_dict, hyperp_opt_result)

    #####################################################
    #   Delete All Suboptimal Trained Neural Networks   #
    #####################################################
    #=== Assigning hyperp with Optimal Hyperparameters ===#
    hyperp_of_interest_list = list(hyperp_of_interest_dict.keys())
    for num, parameter in enumerate(hyperp_of_interest_list):
        setattr(hyperp, parameter, hyperp_opt_result.x[num])

    #=== Updating File Paths with Optimal Hyperparameters ===#
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                                                     project_name,
                                                     data_options, dataset_directory)

    #=== Deleting Suboptimal Neural Networks ===#
    directories_list_trained_NNs = os.listdir(
            path=file_paths.hyperp_opt_trained_NNs_case_directory)
    directories_list_tensorboard = os.listdir(
            path=file_paths.hyperp_opt_tensorboard_case_directory)

    for filename in directories_list_trained_NNs:
        if filename != file_paths.NN_name:
            shutil.rmtree(file_paths.hyperp_opt_trained_NNs_case_directory + '/' + filename)
            shutil.rmtree(file_paths.hyperp_opt_tensorboard_case_directory + '/' + filename)

    print('Suboptimal Trained Networks Deleted')