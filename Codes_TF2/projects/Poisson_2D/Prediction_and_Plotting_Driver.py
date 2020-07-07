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
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty_encoder   = 50
    penalty_decoder   = 0
    penalty_aug       = 50
    penalty_prior     = 0
    batch_size        = 100
    num_epochs        = 1000

class RunOptions:
    def __init__(self):
        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 1
        self.use_reverse_autoencoder = 0

        #=== Autoencoder Loss ===#
        self.use_model_aware = 0
        self.use_model_augmented = 1

        #=== Data Set Size ===#
        self.num_data_train = 5000
        self.num_data_test = 200

        #=== Data Properties ===#
        self.parameter_dimensions = 225
        self.obs_type = 'full'
        self.num_obs_points = 10

        #=== Prior Properties ===#
        self.prior_type_AC = 1
        self.prior_mean_AC = 2
        self.prior_variance_AC = 0.96
        self.prior_corr_AC = 0.002

        #=== Random Seed ===#
        self.random_seed = 1234

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
    if run_options.use_model_aware == 1:
        autoencoder_loss = 'maware_'
    if run_options.use_model_augmented == 1:
        autoencoder_loss = 'maug_'
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(run_options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(run_options.parameter_dimensions)
    file_paths = FilePathsPredictionAndPlotting(hyperp, run_options,
                                   autoencoder_loss, project_name,
                                   data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, run_options, file_paths)
