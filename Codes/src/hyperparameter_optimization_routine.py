#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt import dump, load

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                     Hyperparameter Optimization Routine                     #
###############################################################################
def optimize_hyperparameters(hyperp, options, file_paths,
                             n_calls, space, hyperp_of_interest_dict,
                             data_dict, prior_dict,
                             training_routine, loss_val_index,
                             FilePathsClass, *args):

    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_dict):
        #=== Assign Hyperparameters of Interest ===#
        for key, value in hyperp_of_interest_dict.items():
            hyperp[key] = value

        #=== Update File Paths with New Hyperparameters ===#
        file_paths = FilePathsClass(hyperp, options, *args)

        #=== Training Routine ===#
        training_routine(hyperp, options, file_paths,
                         data_dict, prior_dict)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_array_loss_val = array_metrics[:,loss_val_index]

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
    file_paths = FilePathsClass(hyperp, options, *args)

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

###############################################################################
#                               Output Results                                #
###############################################################################
def output_results(file_paths, hyperp_of_interest_dict, hyperp_opt_result):

    ##################################
    #   Display Optimal Parameters   #
    ##################################
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
    print('Optimized Parameters:')
    hyperp_of_interest_list = list(hyperp_of_interest_dict.keys())
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
