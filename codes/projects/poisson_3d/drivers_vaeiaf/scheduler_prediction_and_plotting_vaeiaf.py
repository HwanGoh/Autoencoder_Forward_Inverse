#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019
@author: Jon Wittmer
"""

import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
import json
import subprocess

from utils_scheduler.get_hyperparameter_combinations import get_hyperparameter_combinations

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Generate Scenarios List                           #
###############################################################################
def generate_scenarios_list():
    hyperp = {}
    hyperp['num_hidden_layers_encoder'] = [5]
    hyperp['num_hidden_layers_decoder'] = [2]
    hyperp['num_hidden_nodes_encoder']  = [1500]
    hyperp['num_hidden_nodes_decoder']  = [500]
    hyperp['activation']                = ['relu']
    hyperp['num_iaf_transforms']        = [10, 20, 30]
    hyperp['num_hidden_nodes_iaf']      = [10]
    hyperp['activation_iaf']            = [10]
    hyperp['penalty_iaf']               = [0.001]
    hyperp['penalty_prior']             = [0.001]
    hyperp['num_data_train']            = [10000]
    hyperp['batch_size']                = [100]
    hyperp['num_epochs']                = [1000]

    return get_hyperparameter_combinations(hyperp)

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    #=== Get list of dictionaries of all combinations ===#
    scenarios_list = generate_scenarios_list()
    print('scenarios_list generated')

    #=== Convert dictionary to json string ===#
    for scenario in scenarios_list:
        scenario_json = json.dumps(scenario)
        proc = subprocess.Popen(
                ['./prediction_and_plotting_driver_vaeiaf.py', f'{scenario_json}'])
        proc.wait()

    print('All scenarios computed')
