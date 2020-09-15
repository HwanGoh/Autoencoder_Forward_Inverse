#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019
@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

import json
from attrdict import AttrDict

# Import FilePaths class and training routine
from Utilities.file_paths_AE import FilePathsTraining
from Utilities.training_routine_custom_AE_model_aware import trainer_custom

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
#                                 Driver                                      #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('json_files/hyperparameters_AE.json') as f:
        hyperp = json.load(f)
    if len(sys.argv) > 1:
        hyperp = command_line_json_string_to_dict(sys.argv, hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('json_files/options_AE.json') as f:
        options = json.load(f)
    options = AttrDict(options)
    options = add_options(options)
    options.model_aware = 1
    options.model_augmented = 0

    #=== File Names ===#
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(options.parameter_dimensions)
    file_paths = FilePathsTraining(hyperp, options,
                                   project_name,
                                   data_options, dataset_directory)

    #=== Initiate training ===#
    trainer_custom(hyperp, options, file_paths)
