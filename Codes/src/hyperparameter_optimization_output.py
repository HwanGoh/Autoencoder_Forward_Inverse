#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import numpy as np
import pandas as pd

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import FilePaths class and training routine
from Utilities.file_paths_AE import FilePathsHyperparameterOptimization

# Import skopt code
from skopt.plots import plot_convergence
from skopt import dump, load

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def output_results(hyperp, run_options, file_paths,
                   hyperp_of_interest_dict, hyperp_opt_result):

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
