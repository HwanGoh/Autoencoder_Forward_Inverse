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

import json
from attrdict import AttrDict

# Import routine for outputting results
from hyperparameter_optimization_output import output_results

# Import FilePaths class and training routine
from Utilities.file_paths_AE import FilePathsHyperparameterOptimization
from Utilities.training_routine_custom_AE_model_augmented_autodiff import\
        trainer_custom

# Import skopt code
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

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
    for key, val in hyperp_of_interest_dict.items():
        space.append(val)

    #=== Instantiate Hyperparameters and Run Options to Load Data ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()
    autoencoder_loss = 'maug_'
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(run_options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(run_options.parameter_dimensions)
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                                   autoencoder_loss, project_name,
                                   data_options, dataset_directory)

    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_dict):
        #=== Assign Hyperparameters of Interest ===#
        for key, val in hyperp_of_interest_dict.items():
            hyperp[key] = val

        #=== Update File Paths with New Hyperparameters ===#
        file_paths = FilePathsHyperparameterOptimization(hyperp, options,
                                                     autoencoder_loss, project_name,
                                                     data_options, dataset_directory)
        #=== Training Routine ===#
        trainer_custom(hyperp, options, file_paths)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_array_loss_val = array_metrics[:,5]

        return storage_array_loss_val[-1]

    ################################
    #   Optimize Hyperparameters   #
    ################################
    hyperp_opt_result = gp_minimize(objective_functional, space,
                                    n_calls=n_calls, random_state=None)

    ######################
    #   Output Results   #
    ######################
    output_results(file_paths, hyperp_of_interest_dict, hyperp_opt_result)

    #####################################################
    #   Delete All Suboptimal Trained Neural Networks   #
    #####################################################
    #=== Assigning hyperp with Optimal Hyperparameters ===#
    for num, key in enumerate(hyperp_of_interest_dict.keys()):
        hyperp[key] = hyperp_opt_result.x[num]

    #=== Updating File Paths with Optimal Hyperparameters ===#
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                                                     autoencoder_loss, project_name,
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
