#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:57:30 2020

@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and plotting routines
from Utilities.file_paths_VAE import FilePathsPlotting
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
    activation        = 'tanh'
    batch_size        = 100
    num_epochs        = 10

class RunOptions:
    def __init__(self):
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

        #=== Posterior Covariance Shape ===#
        self.diagonal_posterior_covariance = 1
        self.full_posterior_covariance = 0

        #=== Matern and Square Exponential Prior Properties ===#
        self.prior_type_nonelliptic = 1
        self.kern_type = 'm32'
        self.prior_cov_length = 0.8

        #=== Elliptic Prior Properties ===#
        self.prior_type_elliptic = 0
        self.prior_type = 'elliptic'
        self.prior_elliptic_d_p = 1
        self.prior_elliptic_g_p = 0.0001

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
        hyperp.batch_size        = int(sys.argv[6])
        hyperp.num_epochs        = int(sys.argv[7])

    #=== File Paths ===#
    if run_options.use_model_aware == 1:
        autoencoder_loss = 'maware_'
    if run_options.use_model_induced == 1:
        autoencoder_loss = 'mind_'
    dataset_directory = '../../../../Datasets/Thermal_Fin/'
    file_paths = FilePathsPlotting(hyperp, run_options,
            autoencoder_loss, dataset_directory)

    #=== Plot and Save Matplotlib ===#
    fig_size = (5,5)
    plot_and_save_predictions(hyperp, run_options, file_paths, fig_size)
    plot_and_save_metrics(hyperp, run_options, file_paths, fig_size)

    #=== Plot and Save vtkfiles ===#
    #plot_and_save_predictions_vtkfiles(hyperp, run_options, file_paths)
