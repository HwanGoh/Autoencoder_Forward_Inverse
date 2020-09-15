#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:51:00 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

import numpy as np
import pandas as pd

import json
from attrdict import AttrDict

# Import routine for outputting results
from hyperparameter_optimization_routine import optimize_hyperparameters

# Import Project Utilities
from Utilities.file_paths_AE import FilePathsHyperparameterOptimization
from Utilities.construct_data_dict import construct_data_dict
from Utilities.construct_prior_dict import construct_prior_dict
from Utilities.training_routine_custom_AE_model_aware import trainer_custom

# Import skopt routines
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
    hyperp_of_interest_dict['penalty_encoder'] = Real(0.01, 50, name='penalty_encoder')
    hyperp_of_interest_dict['penalty_decoder'] = Real(0.01, 50, name='penalty_decoder')
    hyperp_of_interest_dict['penalty_prior'] = Real(0.01, 50, name='penalty_prior')
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
    with open('config_files/hyperparameters_AE.json') as f:
        hyperp = json.load(f)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('config_files/options_AE.json') as f:
        options = json.load(f)
    options = AttrDict(options)
    options = add_options(options)
    options.model_aware = 1
    options.model_augmented = 0

    #=== File Paths ===#
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(options.parameter_dimensions)
    file_paths = FilePathsHyperparameterOptimization(hyperp, options,
                                                     project_name,
                                                     data_options, dataset_directory)
    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, file_paths)
    prior_dict = construct_prior_dict(hyperp, options, file_paths,
                                      load_mean = 1,
                                      load_covariance = 0,
                                      load_covariance_cholesky = 0,
                                      load_covariance_cholesky_inverse = 1)

    ###############################
    #   Optimize Hyperparameters  #
    ###############################
    optimize_hyperparameters(hyperp, options, file_paths,
                             n_calls, space, hyperp_of_interest_dict,
                             data_dict, prior_dict,
                             trainer_custom, 5,
                             FilePathsHyperparameterOptimization,
                             project_name, data_options, dataset_directory)
