#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:37:09 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.insert(0, os.path.realpath('..'))

import yaml
import json
from attrdict import AttrDict

# Import src code
from utils_io.config_io import command_line_json_string_to_dict
from utils_io.file_paths_vae import FilePathsTraining

# Import Project Utilities
from utils_project.file_paths_project import FilePathsProject
from utils_project.construct_data_dict import construct_data_dict
from utils_project.construct_prior_dict import construct_prior_dict
from utils_project.training_routine_custom_vaeiaf_model_aware import trainer_custom

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
#                                    Driver                                   #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_vaeiaf.yaml') as f:
        hyperp = yaml.load(f, Loader=yaml.FullLoader)
    if len(sys.argv) > 1:
        hyperp = command_line_json_string_to_dict(sys.argv, hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_vaeiaf.yaml') as f:
        options = yaml.load(f, Loader=yaml.FullLoader)
    options = AttrDict(options)
    options = add_options(options)
    options.model_aware = 1
    options.model_augmented = 0
    options.posterior_diagonal_covariance = 0
    options.posterior_iaf = 1

    #=== File Paths ===#
    project_paths = FilePathsProject(options)
    file_paths = FilePathsTraining(hyperp, options, project_paths)

    #=== Data and Prior Dictionary ===#
    data_dict = construct_data_dict(hyperp, options, file_paths)
    prior_dict = construct_prior_dict(hyperp, options, file_paths,
                                      load_mean = 1,
                                      load_covariance = 0,
                                      load_covariance_cholesky = 0,
                                      load_covariance_cholesky_inverse = 1)

    #=== Initiate training ===#
    trainer_custom(hyperp, options, file_paths,
                   data_dict, prior_dict)
