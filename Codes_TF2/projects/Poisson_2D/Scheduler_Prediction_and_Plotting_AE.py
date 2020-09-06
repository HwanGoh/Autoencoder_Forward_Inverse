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
    hyperp.num_hidden_layers = [8]
    hyperp.truncation_layer  = [6]
    hyperp.num_hidden_nodes  = [500]
    hyperp.activation        = ['relu']
    hyperp.penalty_encoder   = [10, 50, 100, 1000]
    hyperp.penalty_decoder   = [10]
    hyperp.penalty_aug       = [10]
    hyperp.penalty_prior     = [0.001, 0.01, 0.1, 1]
    hyperp.batch_size        = [100]
    hyperp.num_epochs        = [1000]

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
        proc = subprocess.Popen(['./Prediction_and_Plotting_Driver_AE.py',
            f'{scenario.num_hidden_layers}',
            f'{scenario.truncation_layer}',
            f'{scenario.num_hidden_nodes}',
            f'{scenario.activation}',
            f'{scenario.penalty_encoder:.9f}',
            f'{scenario.penalty_decoder:.9f}',
            f'{scenario.penalty_aug:.9f}',
            f'{scenario.penalty_prior:.9f}',
            f'{scenario.batch_size}',
            f'{scenario.num_epochs}'])
        proc.wait()

    print('All scenarios computed')
