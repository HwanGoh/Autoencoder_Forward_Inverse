#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))
import subprocess
import copy
from get_hyperparameter_permutations import get_hyperparameter_permutations
from Prediction_and_Plotting_Driver_AE import Hyperparameters

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    #########################
    #   Get Scenarios List  #
    #########################
    hyperp = Hyperparameters() # Assign instance attributes below

    # assign instance attributes for hyperp
    hyperp.num_hidden_layers_encoder = [5]
    hyperp.num_hidden_layers_decoder = [2]
    hyperp.num_hidden_nodes_encoder  = [500]
    hyperp.num_hidden_nodes_decoder  = [500]
    hyperp.activation                = ['relu']
    hyperp.num_IAF_transforms        = [1, 5, 10, 20]
    hyperp.num_hidden_nodes_IAF      = [100]
    hyperp.activation_IAF            = ['relu']
    hyperp.penalty_IAF               = [1, 5, 10, 50]
    hyperp.penalty_prior             = [1]
    hyperp.num_data_train            = [10000]
    hyperp.batch_size                = [100]
    hyperp.num_epochs                = [1000]

    permutations_list, hyperp_keys = get_hyperparameter_permutations(hyperp)
    print('permutations_list generated')

    # Convert each list in permutations_list into class attributes
    scenarios_class_instances = []
    for scenario_values in permutations_list:
        hyperp_scenario = Hyperparameters()
        for i in range(0, len(scenario_values)):
            setattr(hyperp_scenario, hyperp_keys[i], scenario_values[i])
        scenarios_class_instances.append(copy.deepcopy(hyperp_scenario))

    for scenario in scenarios_class_instances:
        proc = subprocess.Popen(['./Prediction_and_Plotting_Driver_VAEIAF.py',
            f'{data.num_hidden_layers_encoder}',
            f'{data.num_hidden_layers_decoder}',
            f'{data.num_hidden_nodes_encoder}',
            f'{data.num_hidden_nodes_decoder}',
            f'{scenario.activation}',
            f'{scenario.num_IAF_transforms}',
            f'{scenario.num_hidden_nodes_IAF}',
            f'{scenario.activation_IAF}',
            f'{scenario.penalty_IAF:.9f}',
            f'{scenario.penalty_prior:.9f}',
            f'{scenario.num_data_train}',
            f'{scenario.batch_size}',
            f'{scenario.num_epochs}'])
        proc.wait()

    print('All scenarios computed')
