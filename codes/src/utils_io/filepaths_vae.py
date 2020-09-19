#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os

from utils_io.value_to_string import value_to_string

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 FilePaths                                   #
###############################################################################
class FilePaths():
    def __init__(self, hyperp, options, project_paths):

        ####################
        #   From Project   #
        ####################
        #=== Project ===#
        self.project = project_paths

        #=== Datasets ===#
        self.input_train_file_path = project_paths.input_train_file_path
        self.input_test_file_path = project_paths.input_test_file_path
        self.output_train_file_path = project_paths.output_train_file_path
        self.output_test_file_path = project_paths.output_test_file_path

        #=== Prior ===#
        self.prior_mean_file_path = project_paths.prior_mean_file_path
        self.prior_covariance_file_path = project_paths.prior_covariance_file_path
        self.prior_covariance_cholesky_file_path =\
                project_paths.prior_covariance_cholesky_file_path
        self.prior_covariance_cholesky_inverse_file_path =\
                project_paths.prior_covariance_cholesky_inverse_file_path

        #=== Case Name ===#
        self.case_name = project_paths.case_name

        #=== Neural Network Architecture and Regularization ===#
        autoencoder_type = 'VAE'
        if options.resnet == True:
            resnet = 'res_'
        else:
            resnet = 'nores_'
        if options.model_aware == True:
            forward_model_type = 'maware_'
        if options.model_augmented == True:
            forward_model_type = 'maug_'

        #=== File Name ===#
        if options.posterior_diagonal_covariance == True:
            penalty_kld_incr_string = value_to_string(hyperp.penalty_kld_incr)
            penalty_post_mean_string = value_to_string(hyperp.penalty_post_mean)
            self.NN_name = autoencoder_type + '_' + forward_model_type + resnet +\
                'urg%d_hle%d_hld%d_hne%d_hnd%d_%s_kli%s_klr%d_pm%s_d%d_b%d_e%d' %(
                        options.num_noisy_obs_unregularized,
                        hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                        hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                        hyperp.activation,
                        penalty_kld_incr_string, hyperp.penalty_kld_rate,
                        penalty_post_mean_string,
                        hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        if options.posterior_iaf == True:
            if options.iaf_lstm_update == True:
                iaf_type_string = 'IAFLSTM_'
            else:
                iaf_type_string = 'IAF_'
            penalty_iaf_string = value_to_string(hyperp.penalty_iaf)
            penalty_prior_string = value_to_string(hyperp.penalty_prior)
            self.NN_name = autoencoder_type + iaf_type_string + forward_model_type + resnet +\
                'urg%d_hle%d_hld%d_hne%d_hnd%d_%s_hli%d_hni%d_%s_pi%s_pr%s_d%d_b%d_e%d' %(
                        options.num_noisy_obs_unregularized,
                        hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                        hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                        hyperp.activation,
                        hyperp.num_iaf_transforms, hyperp.num_hidden_nodes_iaf,
                        hyperp.activation_iaf,
                        penalty_iaf_string,
                        penalty_prior_string,
                        hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        #=== Filename ===#
        self.filename = self.case_name + '/' + self.NN_name

###############################################################################
#                               Derived Classes                               #
###############################################################################
class FilePathsTraining(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsTraining, self).__init__(*args, **kwargs)

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = '../../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.NN_name
        self.tensorboard_directory = '../../../../Tensorboard/' + self.filename

class FilePathsHyperparameterOptimization(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsHyperparameterOptimization, self).__init__(*args, **kwargs)

        #=== Parent Directory ===#
        self.hyperp_opt_outputs_directory = 'Hyperparameter_Optimization'

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.NN_savefile_directory = self.hyperp_opt_outputs_directory + '/Trained_NNs/' +\
                self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.NN_name
        self.tensorboard_directory = self.hyperp_opt_outputs_directory + '/Tensorboard/' +\
                self.filename

        #=== For Deleting Suboptimal Networks ===#
        self.hyperp_opt_trained_NNs_case_directory = self.hyperp_opt_outputs_directory +\
                '/Trained_NNs/' + self.case_name
        self.hyperp_opt_tensorboard_case_directory = self.hyperp_opt_outputs_directory +\
                '/Tensorboard/' + self.case_name

        #=== Saving Hyperparameter Optimization Outputs  ===#
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
        self.NN_savefile_directory = '../../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.NN_name

        #=== File Path for Loading Displayable Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test'
        self.savefile_name_state_test = self.NN_savefile_directory + '/state_test'

        #=== File Path for Loading Displayable Predictions ===#
        self.savefile_name_parameter_pred = self.NN_savefile_directory + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_directory + '_state_pred'

        #=== File Path for Saving Figures ===#
        self.figures_savefile_directory = '../../../../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.NN_name
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' +\
                'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory+ '/' +\
                'state_test'
        self.figures_savefile_name_parameter_pred =\
                self.figures_savefile_directory + '/parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_directory + '/state_pred'

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
