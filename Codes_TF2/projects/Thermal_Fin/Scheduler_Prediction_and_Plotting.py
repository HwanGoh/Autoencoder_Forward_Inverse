#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import subprocess
import copy
from Utilities.get_hyperparameter_permutations import get_hyperparameter_permutations
#from Plotting_Driver_Thermal_Fin import Hyperparameters
from Plotting_Driver_Paraview_Thermal_Fin import Hyperparameters
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    #########################
    #   Get Scenarios List  #
    #########################
    hyperp = Hyperparameters() # Assign instance attributes below,
                               # DO NOT assign an instance attribute to GPU

    # assign instance attributes for hyperp
    hyperp.data_type         = ['full','bnd']
    hyperp.num_hidden_layers = [5]
    hyperp.truncation_layer  = [3]
    hyperp.num_hidden_nodes  = [500]
    hyperp.activation        = ['relu']
    hyperp.penalty_encoder   = [0.01, 1, 10, 50]
    hyperp.penalty_decoder   = [0.01, 1, 10, 50]
    hyperp.penalty_aug       = [0.0]
    hyperp.penalty_prior     = [0.0]
    hyperp.batch_size        = [1000]
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
        # proc = subprocess.Popen(['./Prediction_Driver_Thermal_Fin.py',
        #     f'{scenario.data_type}', f'{scenario.num_hidden_layers}',
        #     f'{scenario.truncation_layer}', f'{scenario.num_hidden_nodes}',
        #     f'{scenario.activation}',
        #     f'{scenario.penalty_encoder:.3f}', f'{scenario.penalty_decoder:.3f}',
        #     f'{scenario.penalty_aug:.3f}', f'{scenario.penalty_prior:.3f}',
        #     f'{scenario.batch_size}', f'{scenario.num_epochs}'])
        # proc = subprocess.Popen(['./Plotting_Driver_Thermal_Fin.py',
        #     f'{scenario.data_type}', f'{scenario.num_hidden_layers}',
        #     f'{scenario.truncation_layer}', f'{scenario.num_hidden_nodes}',
        #     f'{scenario.activation}',
        #     f'{scenario.penalty_encoder:.3f}', f'{scenario.penalty_decoder:.3f}',
        #     f'{scenario.penalty_aug:.3f}', f'{scenario.penalty_prior:.3f}',
        #     f'{scenario.batch_size}', f'{scenario.num_epochs}'])
        proc = subprocess.Popen(['./Plotting_Driver_Paraview_Thermal_Fin.py',
            f'{scenario.data_type}', f'{scenario.num_hidden_layers}',
            f'{scenario.truncation_layer}', f'{scenario.num_hidden_nodes}',
            f'{scenario.activation}',
            f'{scenario.penalty_encoder:.3f}', f'{scenario.penalty_decoder:.3f}',
            f'{scenario.penalty_aug:.3f}', f'{scenario.penalty_prior:.3f}',
            f'{scenario.batch_size}', f'{scenario.num_epochs}'])
        proc.wait()

    print('All scenarios computed')