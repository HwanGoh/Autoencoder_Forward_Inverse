#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019
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

# Import FilePaths class and plotting routine
from utils_project.file_paths_ae import FilePathsPredictionAndPlotting
from utils_project.prediction_and_plotting_routine_ae import predict_and_plot, plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    options.model_aware = 0
    options.model_augmented = 1

    return options

###############################################################################
#                                   Driver                                    #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_ae.yaml') as f:
        hyperp = yaml.load(f, Loader=yaml.FullLoader)
    if len(sys.argv) > 1: # if run from scheduler
        hyperp = command_line_json_string_to_dict(sys.argv, hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_ae.yaml') as f:
        options = yaml.load(f, Loader=yaml.FullLoader)
    options = AttrDict(options)
    options = add_options(options)

    #=== File Names ===#
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(options.parameter_dimensions)
    file_paths = FilePathsPredictionAndPlotting(hyperp, options,
                                                project_name,
                                                data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, options, file_paths)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, options, file_paths)
