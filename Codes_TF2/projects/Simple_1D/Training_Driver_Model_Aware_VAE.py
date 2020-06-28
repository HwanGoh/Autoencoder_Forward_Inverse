#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:37:09 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and training routine
from Utilities.file_paths_VAE import FilePathsTraining
from Utilities.training_routine_custom_model_aware_VAE import trainer_custom

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'tanh'
    batch_size        = 100
    num_epochs        = 10

class RunOptions:
    def __init__(self):
        #=== Use Distributed Strategy ===#
        self.use_distributed_training = 0

        #=== Which GPUs to Use for Distributed Strategy ===#
        self.dist_which_gpus = '0,1,2'

        #=== Which Single GPU to Use ===#
        self.which_gpu = '3'

        #=== Data Set Size ===#
        self.num_data_train = 1000
        self.num_data_test = 200

        #=== Posterior Covariance Shape ===#
        self.diagonal_posterior_covariance = 1
        self.full_posterior_covariance = 0

        #=== Prior Properties ===#
        self.diagonal_prior_covariance = 1
        self.full_prior_covariance = 1

        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Data Type ===#
        self.data_type_exponential = 1

        #=== Parameter and Observation Dimensions === #
        self.parameter_dimensions = 2
        self.state_dimensions = 50

###############################################################################
#                                    Driver                                   #
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
        hyperp.batch_size        = int(sys.argv[6])
        hyperp.num_epochs        = int(sys.argv[7])
        run_options.which_gpu    = str(sys.argv[8])

    #=== File Paths ===#
    autoencoder_loss = 'maware_'
    project_name = 'simple_1D_'
    data_options =\
            'm%d' %(run_options.state_dimensions)
    dataset_directory = '../../../../Datasets/Simple_1D/'
    file_paths = FilePathsTraining(hyperp, run_options, autoencoder_loss, project_name,
            data_options, dataset_directory)

    #=== Initiate Training ===#
    trainer_custom(hyperp, run_options, file_paths)
