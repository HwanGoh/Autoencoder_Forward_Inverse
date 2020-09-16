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

from config_utilities import command_line_json_string_to_dict

# Import FilePaths class and plotting routine
from Utilities.file_paths_AE import FilePathsPredictionAndPlotting
from Utilities.prediction_and_plotting_routine_AE import predict_and_plot, plot_and_save_metrics

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

    #=== File Names ===#
    if options.model_aware == 1:
        forward_model_type = 'maware_'
    if options.model_augmented == 1:
        forward_model_type = 'maug_'
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(options.parameter_dimensions)
    file_paths = FilePathsPredictionAndPlotting(hyperp, options,
                                   forward_model_type, project_name,
                                   data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, options, file_paths,
                     project_name, data_options, dataset_directory)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, options, file_paths)
