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

# Import src code
from utils_io.file_paths_ae import FilePathsHyperparameterOptimization
from utils_hyperparameter_optimization.hyperparameter_optimization_routine\
        import optimize_hyperparameters

# Import Project Utilities
from utils_project.file_paths_project import FilePathsProject
from utils_project.construct_data_dict import construct_data_dict
from utils_project.construct_prior_dict import construct_prior_dict
from utils_project.training_routine_custom_ae_model_augmented_autodiff import\
        trainer_custom

# Import skopt code
from skopt.space import Real, Integer, Categorical

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    #=== Use Distributed Strategy ===#
    options.distributed_training = False

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
    hyperp_of_interest_dict['num_hidden_layers_encoder'] = Integer(2, 10,
            name='num_hidden_layers_encoder')
    hyperp_of_interest_dict['num_hidden_nodes_encoder'] = Integer(10, 1000,
            name='num_hidden_nodes_encoder')
    hyperp_of_interest_dict['penalty_encoder'] = Real(10, 1000, name='penalty_encoder')
    hyperp_of_interest_dict['penalty_decoder'] = Real(10, 1000, name='penalty_decoder')
    hyperp_of_interest_dict['penalty_aug'] = Real(0.001, 50, name='penalty_aug')
    hyperp_of_interest_dict['penalty_prior'] = Real(10, 1000, name='penalty_prior')
    #hyperp_of_interest_dict['activation'] = Categorical(['elu', 'relu', 'tanh'], name='activation')
    #hyperp_of_interest_dict['batch_size'] = Integer(100, 500, name='batch_size')

    #####################
    #   Initial Setup   #
    #####################
    #=== Generate skopt 'space' list ===#
    space = []
    for key, value in hyperp_of_interest_dict.items():
        space.append(value)

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_ae.yaml') as f:
        hyperp = yaml.safe_load(f)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_ae.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options = add_options(options)
    options.model_aware = False
    options.model_augmented = True

    #=== File Paths ===#
    project_paths = FilePathsProject(options)
    file_paths = FilePathsHyperparameterOptimization(hyperp, options, project_paths)

    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, file_paths)
    prior_dict = construct_prior_dict(hyperp, options, file_paths,
                                      load_mean = True,
                                      load_covariance = False,
                                      load_covariance_cholesky = False,
                                      load_covariance_cholesky_inverse = True)

    ###############################
    #   Optimize Hyperparameters  #
    ###############################
    optimize_hyperparameters(hyperp, options, file_paths,
                             n_calls, space, hyperp_of_interest_dict,
                             data_dict, prior_dict,
                             trainer_custom, 5,
                             FilePathsHyperparameterOptimization, project_paths)
