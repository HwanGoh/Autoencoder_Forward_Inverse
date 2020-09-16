#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:51:00 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.insert(0, os.path.realpath('..'))

import numpy as np
import pandas as pd

import yaml
from attrdict import AttrDict

# Import routine for outputting results
from utils_hyperparameter_optimization.hyperparameter_optimization_routine\
        import optimize_hyperparameters

# Import Project Utilities
from utils_project.file_paths_vae import FilePathsHyperparameterOptimization
from utils_project.construct_data_dict import construct_data_dict
from utils_project.construct_prior_dict import construct_prior_dict
from utils_project.training_routine_custom_vae_model_aware import trainer_custom

# Import skopt code
from skopt.space import Real, Integer, Categorical

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    #=== Use Distributed Strategy ===#
    options.distributed_training = 0

    #=== Which GPUs to Use for Distributed Strategy ===#
    options.dist_which_gpus = '0,1,2,3'

    #=== Which Single GPU to Use ===#
    options.which_gpu = '2'

    return options

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

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_vae.yaml') as f:
        hyperp = yaml.load(f, Loader=yaml.FullLoader)
    if len(sys.argv) > 1:
        hyperp = command_line_json_string_to_dict(sys.argv, hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_vae.yaml') as f:
        options = yaml.load(f, Loader=yaml.FullLoader)
    options = AttrDict(options)
    options = add_options(options)
    options.model_aware = 1
    options.model_augmented = 0
    options.posterior_diagonal_covariance = 1
    options.posterior_iaf = 0

    #=== File Paths ===#
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(options.parameter_dimensions)
    dataset_directory = '../../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(options.parameter_dimensions)
    file_paths = FilePathsHyperparameterOptimization(hyperp, options,
                                                     project_name,
                                                     data_options, dataset_directory)

    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, file_paths)
    prior_dict = construct_prior_dict(hyperp, options, file_paths,
                                      load_mean = 1,
                                      load_covariance = 1,
                                      load_covariance_cholesky = 0,
                                      load_covariance_cholesky_inverse = 0)

    ###############################
    #   Optimize Hyperparameters  #
    ###############################
    optimize_hyperparameters(hyperp, options, file_paths,
                             n_calls, space, hyperp_of_interest_dict,
                             data_dict, prior_dict,
                             trainer_custom, 5,
                             FilePathsHyperparameterOptimization,
                             project_name, data_options, dataset_directory)