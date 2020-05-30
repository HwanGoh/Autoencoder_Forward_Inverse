#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and plotting routines
from Utilities.file_paths import FilePathsPlotting
from Utilities.plot_and_save_predictions_thermal_fin import plot_and_save_predictions
from Utilities.plot_and_save_predictions_vtkfiles_thermal_fin import plot_and_save_predictions_vtkfiles
from Utilities.plot_and_save_metrics import plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty_encoder   = 50
    penalty_decoder   = 1
    penalty_aug       = 50
    penalty_prior     = 0.0
    batch_size        = 1000
    num_epochs        = 10

class RunOptions:
    def __init__(self):
        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 1
        self.use_reverse_autoencoder = 0

        #=== Autoencoder Loss ===#
        self.use_model_aware = 1
        self.use_model_induced = 0

        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1

        #=== Data Set Size ===#
        self.num_data_train = 200
        self.num_data_test = 200

        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 1
        self.fin_dimensions_3D = 0

        #=== Prior Properties ===#
        if self.fin_dimensions_2D == 1:
            self.kern_type = 'sq_exp'
            self.prior_cov_length = 0.8
            self.prior_mean = 0.0
        if self.fin_dimensions_3D == 1:
            self.kern_type = 'sq_exp'
            self.prior_cov_length = 0.8
            self.prior_mean = 0.0

        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090
        if self.data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if self.data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions

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
        hyperp.penalty_encoder   = float(sys.argv[6])
        hyperp.penalty_decoder   = float(sys.argv[7])
        hyperp.penalty_aug       = float(sys.argv[8])
        hyperp.penalty_prior     = float(sys.argv[9])
        hyperp.batch_size        = int(sys.argv[10])
        hyperp.num_epochs        = int(sys.argv[11])

    #=== File Paths ===#
    if run_options.use_standard_autoencoder == 1:
        autoencoder_type = ''
    if run_options.use_reverse_autoencoder == 1:
        autoencoder_type = 'rev_'
    if run_options.use_model_aware == 1:
        autoencoder_loss = 'maware_'
    if run_options.use_model_induced == 1:
        autoencoder_loss = 'mind_'
    dataset_directory = '../../../../Datasets/Thermal_Fin/'
    file_paths = FilePathsPlotting(hyperp, run_options,
            autoencoder_type, autoencoder_loss, dataset_directory)

    #=== Plot and Save Matplotlib ===#
    fig_size = (5,5)
    plot_and_save_predictions(hyperp, run_options, file_paths, fig_size)
    plot_and_save_metrics(hyperp, run_options, file_paths, fig_size)

    #=== Plot and Save vtkfiles ===#
    # plot_and_save_predictions_vtkfiles(hyperp, run_options, file_paths)
