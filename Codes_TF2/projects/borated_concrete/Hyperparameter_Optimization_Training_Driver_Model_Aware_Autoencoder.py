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
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Import FilePaths class and training routine
from Utilities.file_paths import FilePathsHyperparameterOptimization
from\
Utilities.hyperparameter_optimization_training_routine_custom_model_aware_autoencoder\
        import trainer_custom

# Import skopt code
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
from skopt import dump, load

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty_encoder   = 50
    penalty_decoder   = 1
    penalty_prior     = 0.0
    batch_size        = 1000
    num_epochs        = 2

class RunOptions:
    def __init__(self):
        #=== Use Distributed Strategy ===#
        self.use_distributed_training = 0

        #=== Which GPUs to Use for Distributed Strategy ===#
        self.dist_which_gpus = '0,1,2,3'

        #=== Which Single GPU to Use ===#
        self.which_gpu = '2'

        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 1
        self.use_reverse_autoencoder = 0

        #=== Data Set Size ===#
        self.num_data_train = 2
        self.num_data_test = 2

        #=== Prior Properties ===#
        self.prior_mean = 0.0
        #=== Random Seed ===#

        self.random_seed = 1234

        #=== Data Type ===#
        self.data_type_transport = 0
        self.data_type_diffusion = 0
        self.data_type_discrepancy_additive = 1
        self.data_type_discrepancy_multiplicative = 0

        #=== Shield Locations ===#
        self.locs_left_boundary = 0.5
        self.locs_right_boundary = 2.5
        self.locs_step = 0.5

        #=== Parameter and Observation Dimensions === #
        self.parameter_dimensions = 4
        self.state_dimensions = 1

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
    hyperp_of_interest_dict['num_hidden_layers'] = Integer(5, 10, name='num_hidden_layers')
    hyperp_of_interest_dict['num_hidden_nodes'] = Integer(100, 1000, name='num_hidden_nodes')
    hyperp_of_interest_dict['penalty_encoder'] = Real(0.01, 50, name='penalty_encoder')
    hyperp_of_interest_dict['penalty_decoder'] = Real(0.01, 50, name='penalty_decoder')
    #hyperp_of_interest_dict['activation'] = Categorical(['elu', 'relu', 'tanh'], name='activation')
    #hyperp_of_interest_dict['batch_size'] = Integer(100, 500, name='batch_size')

    #####################
    #   Initial Setup   #
    #####################
    hyperp_of_interest_list = list(hyperp_of_interest_dict.keys())
    hyperp_of_interest_objective_args_tuple = tuple(hyperp_of_interest_list)

    #=== Generate skopt 'space' list ===#
    space = []
    for key, val in hyperp_of_interest_dict.items():
        space.append(val)

    #=== Instantiate Hyperparameters and Run Options to Load Data ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()
    autoencoder_loss = 'maware_'
    project_name = 'borated_concrete_'
    data_options = 'shl%d_shr%d_shs%d'%(run_options.locs_left_boundary,
            run_options.locs_right_boundary, run_options.locs_step)
    dataset_directory = '../../../../Datasets/Neutron_Transport/borated_concrete/'
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
            autoencoder_loss, project_name,
            data_options, dataset_directory)

    ################
    #   Training   #
    ################
    hyperp_opt_result = trainer_custom(hyperp, run_options, file_paths,
                                       n_calls, space,
                                       autoencoder_loss,
                                       project_name, data_options, dataset_directory)

    ##################################
    #   Display Optimal Parameters   #
    ##################################
    print('=================================================')
    print('      Hyperparameter Optimization Complete')
    print('=================================================')
    print('Optimized Validation Loss: {}\n'.format(hyperp_opt_result.fun))
    print('Optimized Parameters:')
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

    #####################################################
    #   Delete All Suboptimal Trained Neural Networks   #
    #####################################################
    #=== Assigning hyperp with Optimal Hyperparameters ===#
    for num, parameter in enumerate(hyperp_of_interest_list):
        setattr(hyperp, parameter, hyperp_opt_result.x[num])
    hyperp.truncation_layer = int(np.ceil(hyperp.num_hidden_layers/2))

    #=== Updating File Paths with Optimal Hyperparameters ===#
    file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
            autoencoder_loss, project_name, data_options, dataset_directory)

    #=== Deleting Suboptimal Neural Networks ===#
    directories_list_Trained_NNs = os.listdir(path=file_paths.hyperp_opt_Trained_NNs_directory)
    directories_list_Tensorboard = os.listdir(path=file_paths.hyperp_opt_Tensorboard_directory)
    for filename in directories_list_Trained_NNs:
        if filename != file_paths.filename:
            shutil.rmtree(file_paths.hyperp_opt_Trained_NNs_directory + '/' + filename)
            shutil.rmtree(file_paths.hyperp_opt_Tensorboard_directory + '/' + filename)

    print('Suboptimal Trained Networks Deleted')