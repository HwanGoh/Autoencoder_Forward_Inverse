#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

# Import FilePaths class and training routine
from Utilities.file_paths import FilePathsTraining
from Utilities.training_routine_custom_AE_model_augmented_autodiff import\
        trainer_custom
from Utilities.test_gradient_poisson_2D_FEM import test_gradient
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class Hyperparameters:
    num_hidden_layers = 8
    truncation_layer  = 6 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'relu'
    penalty_encoder   = 1
    penalty_decoder   = 1
    penalty_aug       = 1
    penalty_prior     = 0.5
    batch_size        = 10
    num_epochs        = 3

class RunOptions:
    def __init__(self):
        #=== Use Distributed Strategy ===#
        self.distributed_training = 0

        #=== Which GPUs to Use for Distributed Strategy ===#
        self.dist_which_gpus = '0,1,2,3'

        #=== Which Single GPU to Use ===#
        self.which_gpu = '2'

        #=== Autoencoder Type ===#
        self.standard_autoencoder = 0
        self.reverse_autoencoder = 1

        #=== Data Set Size ===#
        self.num_data_train = 5000
        self.num_data_test = 200

        #=== Data Properties ===#
        self.parameter_dimensions = 225
        self.obs_type = 'obs'
        self.num_obs_points = 43

        #=== Noise Properties ===#
        self.add_noise = 1
        self.noise_level = 0.05
        self.num_noisy_obs = 20
        self.num_noisy_obs_unregularized = 20

        #=== Autocorrelation Prior Properties ===#
        self.prior_type_AC_train = 1
        self.prior_mean_AC_train = 2
        self.prior_variance_AC_train = 2.0
        self.prior_corr_AC_train = 0.5

        self.prior_type_AC_test = 1
        self.prior_mean_AC_test = 2
        self.prior_variance_AC_test = 2.0
        self.prior_corr_AC_test = 0.5

        #=== Matern Prior Properties ===#
        self.prior_type_matern_train = 0
        self.prior_kern_type_train = 'm32'
        self.prior_cov_length_train = 0.5

        self.prior_type_matern_test = 0
        self.prior_kern_type_test = 'm32'
        self.prior_cov_length_test = 0.5

        #=== PDE Properties ===#
        self.boundary_matrix_constant = 0.5
        self.load_vector_constant = -1

        #=== Random Seed ===#
        self.random_seed = 4

###############################################################################
#                                    Driver                                   #
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
        hyperp.penalty_aug       = float(sys.argv[7])
        hyperp.penalty_prior     = float(sys.argv[8])
        hyperp.batch_size        = int(sys.argv[9])
        hyperp.num_epochs        = int(sys.argv[10])
        run_options.which_gpu    = str(sys.argv[11])

    #=== File Paths ===#
    autoencoder_loss = 'maug_'
    project_name = 'poisson_2D_'
    data_options = 'n%d' %(run_options.parameter_dimensions)
    dataset_directory = '../../../../Datasets/Finite_Element_Method/Poisson_2D/' +\
            'n%d/'%(run_options.parameter_dimensions)
    file_paths = FilePathsTraining(hyperp, run_options,
                                   autoencoder_loss, project_name,
                                   data_options, dataset_directory)

    #=== Test Gradient ===#
    test_gradient(hyperp, run_options, file_paths)

    #=== Initiate Training ===#
    trainer_custom(hyperp, run_options, file_paths)
