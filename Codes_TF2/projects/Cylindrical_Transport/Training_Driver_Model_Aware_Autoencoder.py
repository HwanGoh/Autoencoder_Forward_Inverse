#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019
@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and training routine
from Utilities.file_paths import FilePathsTraining
from Utilities.training_routine_custom import trainer_custom

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
    num_epochs        = 10

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
        self.num_data_train = 2959
        self.num_data_test = 200

        #=== Prior Properties ===#
        self.prior_mean = 0.0

        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Data Type ===#
        self.data_type_transport = 0
        self.data_type_diffusion = 0
        self.data_type_discrepancy_additive = 0
        self.data_type_discrepancy_multiplicative = 1

        #=== Parameter and Observation Dimensions === #
        self.parameter_dimensions = 3
        self.state_dimensions = 1440

###############################################################################
#                                 Driver                                      #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters and Run Options ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()

    if len(sys.argv) > 1:
        hyperp.data_type         = str(sys.argv[1])
        hyperp.num_hidden_layers = int(sys.argv[2])
        hyperp.truncation_layer  = int(sys.argv[3])
        hyperp.num_hidden_nodes  = int(sys.argv[4])
        hyperp.activation        = str(sys.argv[5])
        hyperp.penalty_encoder   = float(sys.argv[6])
        hyperp.penalty_decoder   = float(sys.argv[7])
        hyperp.penalty_prior     = float(sys.argv[8])
        hyperp.batch_size        = int(sys.argv[9])
        hyperp.num_epochs        = int(sys.argv[10])
        run_options.which_gpu    = str(sys.argv[11])

    #=== File Names ===#
    autoencoder_loss = 'maware_'
    project_name = 'cylindrical_'
    data_options =\
            'n%d' %(run_options.state_dimensions)
    dataset_directory = '../../../../Datasets/Neutron_Transport/cylindrical/'
    file_paths = FilePathsTraining(hyperp, run_options, autoencoder_loss, project_name,
            data_options, dataset_directory)

    #=== Initiate training ===#
    trainer_custom(hyperp, run_options, file_paths)
