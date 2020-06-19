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
    data_type         = 'full'
    num_hidden_layers = 10 # For this architecture, need at least 2. One for the mapping to the
                          # feature space, one as a trainable hidden layer.
                          # EXCLUDES MAPPING BACK TO DATA SPACE
    num_hidden_nodes  = 100
    activation        = 'relu'
    regularization    = 0.001
    batch_size        = 200
    num_epochs        = 1000

class RunOptions:
    def __init__(self):

        #=== Which API ===#
        self.use_tf_custom = 1
        self.use_tf_keras = 0

        #=== Use ResNet ===#
        self.use_resnet = 1

        #=== Use Regularization ===#
        self.use_L1 = 0
        self.use_L2 = 1

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
        hyperp.data_type         = str(sys.argv[1])
        hyperp.num_hidden_layers = int(sys.argv[2])
        hyperp.num_hidden_nodes  = int(sys.argv[3])
        hyperp.activation        = str(sys.argv[4])
        hyperp.regularization    = float(sys.argv[5])
        hyperp.batch_size        = int(sys.argv[6])
        hyperp.num_epochs        = int(sys.argv[7])

    #=== File Names ===#
    NN_type = 'FC_'
    project_name = 'borated_concrete_'
    data_options = 'shl%d_shr%d_shs%d'%(run_options.locs_left_boundary,
            run_options.locs_right_boundary, run_options.locs_step)
    dataset_directory = '../../../../Datasets/Neutron_Transport/borated_concrete/'
    file_paths = FilePathsPredictionAndPlotting(hyperp, run_options, NN_type, project_name,
            data_options, dataset_directory)

    #=== Predict and Save ===#
    predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory)

    #=== Plot and Save ===#
    plot_and_save_metrics(hyperp, run_options, file_paths)
