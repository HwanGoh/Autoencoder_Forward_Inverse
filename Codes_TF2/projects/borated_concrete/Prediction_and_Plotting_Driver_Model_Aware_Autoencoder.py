#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:41:12 2019
@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and plotting routine
from Utilities.file_paths import FilePathsPredictionAndPlotting
from Utilities.prediction_and_plotting_routine import predict_and_plot, plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    num_hidden_layers = 10
    truncation_layer  = 5 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 1000
    activation        = 'relu'
    penalty_encoder   = 0.01
    penalty_decoder   = 0.01
    penalty_prior     = 0.0
    batch_size        = 100
    num_epochs        = 1000

class RunOptions:
    def __init__(self):

        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 0
        self.use_reverse_autoencoder = 1

        #=== Data Set Size ===#
        self.num_data_train = 36000
        self.num_data_test = 200

        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Data Type ===#
        self.data_type_transport = 0
        self.data_type_diffusion = 0
        self.data_type_discrepancy_additive = 0
        self.data_type_discrepancy_multiplicative = 1

        #=== Shield Locations ===#
        self.locs_left_boundary = 0.5
        self.locs_right_boundary = 2.5
        self.locs_step = 0.5

        #=== Parameter and Observation Dimensions === #
        self.parameter_dimensions = 4
        self.state_dimensions = 1

###############################################################################
#                                 Driver                                      #
###############################################################################
if __name__ == "__main__":

    #=== Hyperparameters and Run Options ===#
    hyperp = Hyperparameters()
    run_options = RunOptions()

    if len(sys.argv) > 1:
        hyperp.num_hidden_layers = int(sys.argv[1])
        hyperp.truncation_layer  = int(sys.argv[2])
        hyperp.num_hidden_nodes  = int(sys.argv[3])
        hyperp.activation        = str(sys.argv[4])
        hyperp.penalty_encoder   = float(sys.argv[5])
        hyperp.penalty_decoder   = float(sys.argv[6])
        hyperp.penalty_prior     = float(sys.argv[7])
        hyperp.batch_size        = int(sys.argv[8])
        hyperp.num_epochs        = int(sys.argv[9])

    #=== File Names ===#
    autoencoder_loss = 'maware_'
    project_name = 'borated_concrete_'
    data_options = 'shl%d_shr%d_shs%d'%(run_options.locs_left_boundary,
            run_options.locs_right_boundary, run_options.locs_step)
    dataset_directory = '../../../../Datasets/Neutron_Transport/borated_concrete/'
    file_paths = FilePathsPredictionAndPlotting(hyperp, run_options, autoencoder_loss, project_name,
            data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, run_options, file_paths)