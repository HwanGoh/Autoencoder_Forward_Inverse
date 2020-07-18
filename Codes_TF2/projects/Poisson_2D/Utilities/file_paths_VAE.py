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
#                                 FilePaths                                   #
###############################################################################
class FilePaths():
    def __init__(self, hyperp, run_options,
            autoencoder_loss, project_name,
            data_options, dataset_directory):
        #################
        #   File Name   #
        #################
        #=== Data Type ===#
        if run_options.obs_type == 'full':
            obs_string = 'full'
        if run_options.obs_type == 'obs':
            obs_string = 'obs_o%d'%(run_options.num_obs_points)
        data_string = data_options + '_' + obs_string + '_'

        #=== Prior Properties ===#
        if run_options.prior_type_AC == 1:
            prior_type = 'AC'
            prior_mean = run_options.prior_mean_AC
            prior_variance = run_options.prior_variance_AC
            prior_corr = run_options.prior_corr_AC

        if prior_mean >= 1:
            prior_mean = int(prior_mean)
            prior_mean_string = str(prior_mean)
        else:
            prior_mean_string = str(prior_mean)
            prior_mean_string = 'pt' + prior_mean_string[2:]
        if prior_variance >= 1:
            prior_variance = int(prior_variance)
            prior_variance_string = str(prior_variance)
        else:
            prior_variance_string = str(prior_variance)
            prior_variance_string = 'pt' + prior_variance_string[2:]
        if prior_corr >= 1:
            prior_corr = int(prior_corr)
            prior_corr_string = str(prior_corr)
        else:
            prior_corr_string = str(prior_corr)
            prior_corr_string = 'pt' + prior_corr_string[2:]
        prior_string = '%s_%s_%s_%s'%(prior_type, prior_mean_string,
                prior_variance_string, prior_corr_string)

        #=== Neural Network Architecture and Regularization ===#
        autoencoder_type = 'VAE_'
        if run_options.diagonal_posterior_covariance == 1:
            posterior_covariance_shape = 'diagpost_'
        if run_options.full_posterior_covariance == 1:
            posterior_covariance_shape = 'fullpost_'

        #=== File Name ===#
        self.filename = project_name +\
            data_string + prior_string + '_' +\
            autoencoder_type + autoencoder_loss +\
            'hl%d_tl%d_hn%d_%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers,
                    hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation,
                    run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.obs_indices_savefilepath = dataset_directory +\
                project_name + 'obs_indices_' +\
                'o%d_'%(run_options.num_obs_points) + data_options
        self.input_train_savefilepath = dataset_directory +\
                project_name +\
                'parameter_train_' +\
                'd%d_'%(run_options.num_data_train) + data_options + '_' + prior_string
        self.input_test_savefilepath = dataset_directory +\
                project_name +\
                'parameter_test_' +\
                'd%d_'%(run_options.num_data_test) + data_options + '_' + prior_string
        if run_options.obs_type == 'full':
            self.output_train_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_train_' +\
                    'd%d_'%(run_options.num_data_train) + data_options + '_' + prior_string
            self.output_test_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_test_' +\
                    'd%d_'%(run_options.num_data_test) + data_options + '_' + prior_string
        if run_options.obs_type == 'obs':
            self.output_train_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_train_' +\
                    'o%d_d%d_' %(run_options.num_obs_points, run_options.num_data_train) +\
                    data_options + '_' + prior_string
            self.output_test_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_test_' +\
                    'o%d_d%d_' %(run_options.num_obs_points, run_options.num_data_test) +\
                    data_options + '_' + prior_string

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_mean_savefilepath = dataset_directory +\
                'prior_mean_' + data_options + '_' + prior_string
        self.prior_covariance_savefilepath = dataset_directory +\
                'prior_covariance_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky_savefilepath = dataset_directory +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string
        self.prior_covariance_inverse_savefilepath = dataset_directory +\
                'prior_covariance_inverse_' + data_options + '_' + prior_string

        ###################
        #   FEM Objects   #
        ###################
        #=== Pre-Matrices ===#
        self.premass_savefilepath = dataset_directory +\
                'premass_' + data_options
        self.prestiffness_savefilepath = dataset_directory +\
                'prestiffness_' + data_options
        self.boundary_matrix_savefilepath = dataset_directory +\
                'boundary_matrix_' + data_options
        self.load_vector_savefilepath = dataset_directory +\
                'load_vector_' + data_options

###############################################################################
#                               Derived Classes                               #
###############################################################################
class FilePathsTraining(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsTraining, self).__init__(*args, **kwargs)
        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = '../../../Tensorboard/' + self.filename

class FilePathsHyperparameterOptimization(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsHyperparameterOptimization, self).__init__(*args, **kwargs)
        #=== Saving Trained Neural Network and Tensorboard ===#
        self.hyperp_opt_Trained_NNs_directory = 'Hyperparameter_Optimization/Trained_NNs'
        self.hyperp_opt_Tensorboard_directory = 'Hyperparameter_Optimization/Tensorboard'
        self.NN_savefile_directory = self.hyperp_opt_Trained_NNs_directory + '/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = self.hyperp_opt_Tensorboard_directory + '/' + self.filename

        #=== Saving Hyperparameter Optimization Outputs  ===#
        self.hyperp_opt_outputs_directory = 'Hyperparameter_Optimization'
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

class FilePathsPredictionAndPlotting(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPredictionAndPlotting, self).__init__(*args, **kwargs)
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
###############################################################################
#                               Derived Classes                               #
###############################################################################
class FilePathsTraining(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsTraining, self).__init__(*args, **kwargs)
        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = '../../../Tensorboard/' + self.filename

class FilePathsHyperparameterOptimization(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsHyperparameterOptimization, self).__init__(*args, **kwargs)
        #=== Saving Trained Neural Network and Tensorboard ===#
        self.hyperp_opt_Trained_NNs_directory = 'Hyperparameter_Optimization/Trained_NNs'
        self.hyperp_opt_Tensorboard_directory = 'Hyperparameter_Optimization/Tensorboard'
        self.NN_savefile_directory = self.hyperp_opt_Trained_NNs_directory + '/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.tensorboard_directory = self.hyperp_opt_Tensorboard_directory + '/' + self.filename

        #=== Saving Hyperparameter Optimization Outputs  ===#
        self.hyperp_opt_outputs_directory = 'Hyperparameter_Optimization'
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

class FilePathsPredictionAndPlotting(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPredictionAndPlotting, self).__init__(*args, **kwargs)
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
