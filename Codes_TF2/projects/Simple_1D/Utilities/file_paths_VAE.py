#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os
import time
from decimal import Decimal # for filenames
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 File Name                                   #
###############################################################################
def filename(hyperp, run_options, autoencoder_loss):
    #=== Data Type ===#
    if run_options.data_type_exponential == 1:
        data_type = 'exponential_'
        directory_name = 'Exponential/'

    data_string = data_type

    #=== Neural Network Architecture and Regularization ===#
    autoencoder_type = 'VAE_'
    if run_options.diagonal_posterior_covariance == 1:
        posterior_covariance_shape = 'diagpost_'
    if run_options.full_posterior_covariance == 1:
        posterior_covariance_shape = 'fullpost_'
    if run_options.diagonal_prior_covariance == 1:
        prior_string = '_prdiag'
    if run_options.full_prior_covariance == 1:
        prior_string = '_prfull'

    #=== File Name ===#
    filename = autoencoder_type + posterior_covariance_shape + autoencoder_loss +\
            data_string +\
            prior_string +\
            '_hl%d_tl%d_hn%d_%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers,
                    hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation,
                    run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)

    return data_type, directory_name, prior_string, filename

###############################################################################
#                          Train and Test Datasets                            #
###############################################################################
def train_and_test_datasets(run_options, project_name,
        data_options, dataset_directory, directory_name, data_type):

    input_train_savefilepath =\
        dataset_directory + directory_name + project_name +\
        'parameter_' + data_type + 'train_d%d_'%(run_options.num_data_train) +\
        data_options
    output_train_savefilepath =\
        dataset_directory + directory_name + project_name +\
        'state_' + data_type + 'train_d%d_'%(run_options.num_data_train) +\
        data_options
    input_test_savefilepath =\
        dataset_directory + directory_name + project_name +\
        'parameter_' + data_type + 'test_d%d_'%(run_options.num_data_test) +\
        data_options
    output_test_savefilepath =\
        dataset_directory + directory_name + project_name +\
        'state_' + data_type + 'test_d%d_'%(run_options.num_data_test) +\
        data_options

    return input_train_savefilepath, output_train_savefilepath,\
            input_test_savefilepath, output_test_savefilepath

###############################################################################
#                                 Training                                    #
###############################################################################
class FilePathsTraining():
    def __init__(self, hyperp, run_options,
            autoencoder_loss, project_name,
            data_options, dataset_directory):

        #=== File name ===#
        data_type, directory_name, prior_string, self.filename\
                = filename(hyperp, run_options, autoencoder_loss)

        #=== Loading and saving data ===#
        self.input_train_savefilepath, self.output_train_savefilepath,\
        self.input_test_savefilepath, self.output_test_savefilepath\
                = train_and_test_datasets(run_options, project_name,
                        data_options, dataset_directory, directory_name, data_type)

        #=== Prior File Name ===#
        if run_options.diagonal_prior_covariance == 1:
            prior_string = 'diag_'
        if run_options.full_prior_covariance == 1:
            prior_string = 'full_'
        self.prior_mean_file_name = project_name + 'prior_mean_' +\
                data_type + 'n%d'%(run_options.parameter_dimensions)
        self.prior_mean_savefilepath = dataset_directory + directory_name +\
                self.prior_mean_file_name
        self.prior_covariance_file_name = project_name + 'prior_covariance_' +\
                prior_string + data_type + 'n%d'%(run_options.parameter_dimensions)
        self.prior_covariance_savefilepath = dataset_directory + directory_name +\
                self.prior_covariance_file_name

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = '../../../Tensorboard/' + self.filename

###############################################################################
#                         Hyperparemeter Optimization                         #
###############################################################################
class FilePathsHyperparameterOptimization():
    def __init__(self, hyperp, run_options, autoencoder_loss, dataset_directory):

        #=== File name ===#
        data_type, directory_name, prior_string, self.filename\
                = filename(hyperp, run_options, autoencoder_loss)

        #=== Loading and saving data ===#
        self.input_train_savefilepath, self.output_train_savefilepath,\
        self.input_test_savefilepath, self.output_test_savefilepath\
                = train_and_test_datasets(run_options, project_name,
                        data_options, dataset_directory, directory_name, data_type)

        #=== Prior File Name ===#
        if run_options.diagonal_prior_covariance == 1:
            prior_string = 'diag_'
        if run_options.full_prior_covariance == 1:
            prior_string = 'full_'
        self.prior_cov_file_name = project_name + 'prior_covariance_' +\
                prior_string + data_type + 'n%d'%(run_options.parameter_dimensions)
        self.prior_savefilepath = dataset_directory + directory_name + self.prior_cov_file_name

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.hyperp_opt_Trained_NNs_directory = '../../../Hyperparameter_Optimization/Trained_NNs'
        self.hyperp_opt_Tensorboard_directory = '../../../Hyperparameter_Optimization/Tensorboard'
        self.NN_savefile_directory = self.hyperp_opt_Trained_NNs_directory + '/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = self.hyperp_opt_Tensorboard_directory + '/' + self.filename

        #=== Saving Hyperparameter Optimization Outputs  ===#
        self.hyperp_opt_outputs_directory = '../../../Hyperparameter_Optimization'
        self.hyperp_opt_skopt_res_savefile_name = self.hyperp_opt_outputs_directory +\
                '/hyperp_opt_result.pkl'
        self.hyperp_opt_optimal_parameters_savefile_name = self.hyperp_opt_outputs_directory +\
                '/optimal_set_of_hyperparameters.txt'
        self.hyperp_opt_scenarios_trained_savefile_name = self.hyperp_opt_outputs_directory +\
                '/scenarios_trained.txt'
        self.hyperp_opt_validation_losses_savefile_name = self.hyperp_opt_outputs_directory +\
                '/validation_losses.csv'
        self.hyperp_opt_convergence_savefile_name = self.hyperp_opt_outputs_directory +\
                '/convergence.png'

###############################################################################
#                                 Plotting                                    #
###############################################################################
class FilePathsPredictionAndPlotting():
    def __init__(self, hyperp, run_options,
            autoencoder_loss,
            project_name, data_options,
            dataset_directory):

        #=== File name ===#
        data_type, directory_name, prior_string, self.filename\
                = filename(hyperp, run_options, autoencoder_loss)

        #=== Loading and saving data ===#
        _, _,\
        self.input_test_savefilepath, self.output_test_savefilepath\
                = train_and_test_datasets(run_options, project_name,
                        data_options, dataset_directory, directory_name, data_type)

        #=== File Path for Loading Trained Neural Network ===#
        self.NN_savefile_directory = '../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename

        #=== File Path for Loading Displayable Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory +\
                '/parameter_test'
        self.savefile_name_state_test = self.NN_savefile_directory +\
                '/state_test'

        #=== File Path for Loading Displayable Predictions ===#
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'

        #=== File Path for Saving Figures ===#
        self.figures_savefile_directory = '../../../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' +\
                'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory+ '/' +\
                'state_test'
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
