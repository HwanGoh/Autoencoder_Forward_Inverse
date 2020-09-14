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
from Utilities.file_paths_VAE import FilePathsPredictionAndPlotting
from Utilities.prediction_and_plotting_routine_VAE import predict_and_plot, plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class Hyperparameters:
    num_hidden_layers = 8
    truncation_layer  = 6 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty_KLD_incr  = 1000
    penalty_KLD_rate  = 250
    penalty_post_mean = 1
    num_data_train    = 10000
    batch_size        = 100
    num_epochs        = 1000

class RunOptions:
    #=== Posterior Type ===#
    posterior_diagonal_covariance = 1
    posterior_IAF = 0

    #=== Forward Model Type ===#
    model_aware = 1
    model_augmented = 0

    #=== Use Resnet ===#
    resnet = 0

    #=== Data Set Size ===#
    num_data_train_load = 10000
    num_data_test_load = 200
    num_data_test = 200

    #=== Data Properties ===#
    parameter_dimensions = 225
    obs_type = 'full'
    num_obs_points = 43

    #=== Noise Properties ===#
    add_noise = 0
    noise_level = 0.05
    num_noisy_obs = 20
    num_noisy_obs_unregularized = 20

    #=== Autocorrelation Prior Properties ===#
    prior_type_AC_train = 1
    prior_mean_AC_train = 2
    prior_variance_AC_train = 2.0
    prior_corr_AC_train = 0.5

    prior_type_AC_test = 1
    prior_mean_AC_test = 2
    prior_variance_AC_test = 2.0
    prior_corr_AC_test = 0.5

    #=== Matern Prior Properties ===#
    prior_type_matern_train = 0
    prior_kern_type_train = 'm32'
    prior_cov_length_train = 0.5

    prior_type_matern_test = 0
    prior_kern_type_test = 'm32'
    prior_cov_length_test = 0.5

    #=== Random Seed ===#
    random_seed = 4

###############################################################################
#                                   Driver                                    #
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
        hyperp.penalty_KLD_incr  = float(sys.argv[5])
        hyperp.penalty_KLD_rate  = int(sys.argv[6])
        hyperp.penalty_post_mean = float(sys.argv[7])
        hyperp.num_data_train    = int(sys.argv[8])
        hyperp.batch_size        = int(sys.argv[9])
        hyperp.num_epochs        = int(sys.argv[10])

    #=== File Names ===#
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(run_options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(run_options.parameter_dimensions)
    file_paths = FilePathsPredictionAndPlotting(hyperp, run_options,
                                                project_name,
                                                data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, run_options, file_paths)
