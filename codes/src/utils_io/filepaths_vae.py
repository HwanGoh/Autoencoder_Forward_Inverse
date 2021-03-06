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
        self.input_train = project_paths.input_train
        self.input_test = project_paths.input_test
        self.input_specific = project_paths.input_specific
        self.output_train = project_paths.output_train
        self.output_test = project_paths.output_test
        self.output_specific = project_paths.output_specific

        #=== Prior ===#
        self.prior_string_reg = project_paths.prior_string_reg
        self.prior_mean = project_paths.prior_mean
        self.prior_covariance = project_paths.prior_covariance
        self.prior_covariance_cholesky = project_paths.prior_covariance_cholesky
        self.prior_covariance_cholesky_inverse = project_paths.prior_covariance_cholesky_inverse

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
        penalty_js_string = value_to_string(hyperp.penalty_js)
        if options.posterior_diagonal_covariance == True:
            self.NN_name = autoencoder_type + '_' + forward_model_type + resnet +\
                self.prior_string_reg +\
                '_urg%d_hle%d_hld%d_hne%d_hnd%d_%s_pjs%s_d%d_b%d_e%d' %(
                        options.num_noisy_obs_unregularized,
                        hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                        hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                        hyperp.activation,
                        penalty_js_string,
                        hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        if options.posterior_iaf == True:
            if options.iaf_lstm_update == True:
                iaf_type_string = 'IAFLSTM_'
            else:
                iaf_type_string = 'IAF_'
            self.NN_name = autoencoder_type + iaf_type_string + forward_model_type + resnet +\
                self.prior_string_reg +\
                '_urg%d_hle%d_hld%d_hne%d_hnd%d_%s_hli%d_hni%d_%s_pjs%s_d%d_b%d_e%d'\
                %(options.num_noisy_obs_unregularized,
                  hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                  hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                  hyperp.activation,
                  hyperp.num_iaf_transforms, hyperp.num_hidden_nodes_iaf,
                  hyperp.activation_iaf,
                  penalty_js_string,
                  hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        #=== Filename ===#
        self.case_and_NN_name = self.case_name + '/' + self.NN_name

###############################################################################
#                               Derived Classes                               #
###############################################################################
class FilePathsTraining(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsTraining, self).__init__(*args, **kwargs)

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.directory_trained_NN = '../../../../Trained_NNs/' + self.case_and_NN_name
        self.trained_NN = self.directory_trained_NN + '/' + self.NN_name
        self.directory_tensorboard = '../../../../Tensorboard/' + self.case_and_NN_name

class FilePathsHyperparameterOptimization(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsHyperparameterOptimization, self).__init__(*args, **kwargs)

        #=== Parent Directory ===#
        self.directory_hyperp_opt = '../../../../Hyperparameter_Optimization'

        #=== Saving Trained Neural Network and Tensorboard ===#
        self.directory_trained_NN = self.directory_hyperp_opt + '/Trained_NNs/' +\
                self.case_and_NN_name
        self.trained_NN = self.directory_trained_NN + '/' + self.NN_name
        self.directory_tensorboard = self.directory_hyperp_opt + '/Tensorboard/' +\
                self.case_and_NN_name

        #=== Saving Hyperparameter Optimization Outputs  ===#
        self.directory_hyperp_opt_outputs = self.directory_hyperp_opt + '/Outputs/' +\
                self.case_name
        self.hyperp_opt_skopt_res = self.directory_hyperp_opt_outputs +\
                '/hyperp_opt_result.pkl'
        self.hyperp_opt_optimal_parameters = self.directory_hyperp_opt_outputs +\
                '/optimal_set_of_hyperparameters.txt'
        self.hyperp_opt_scenarios_trained = self.directory_hyperp_opt_outputs +\
                '/scenarios_trained.txt'
        self.hyperp_opt_validation_losses = self.directory_hyperp_opt_outputs +\
                '/validation_losses.csv'
        self.hyperp_opt_convergence = self.directory_hyperp_opt_outputs +\
                '/convergence.png'

        #=== For Deleting Suboptimal Networks ===#
        self.directory_hyperp_opt_trained_NN_case = self.directory_hyperp_opt +\
                '/Trained_NNs/' + self.case_name
        self.directory_hyperp_opt_tensorboard_case = self.directory_hyperp_opt +\
                '/Tensorboard/' + self.case_name

class FilePathsPredictionAndPlotting(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPredictionAndPlotting, self).__init__(*args, **kwargs)

        #=== File Path for Loading Trained Neural Network ===#
        self.directory_trained_NN = '../../../../Trained_NNs/' + self.case_and_NN_name
        self.trained_NN = self.directory_trained_NN + '/' + self.NN_name

        #=== File Path for Loading Displayable Test Data ===#
        self.parameter_test = self.directory_trained_NN + '/parameter_test'
        self.state_test = self.directory_trained_NN + '/state_test'

        #=== File Path for Loading Displayable Predictions ===#
        self.parameter_pred = self.directory_trained_NN + '_parameter_pred'
        self.state_pred = self.directory_trained_NN + '_state_pred'

        #=== File Path for Saving Figures ===#
        self.directory_figures = '../../../../Figures/' + self.case_and_NN_name
        self.figures = self.directory_figures + '/' + self.NN_name
        self.figure_parameter_test = self.directory_figures + '/parameter_test'
        self.figure_state_test = self.directory_figures+ '/state_test'
        self.figure_parameter_pred = self.directory_figures + '/parameter_pred'
        self.figure_state_pred = self.directory_figures + '/state_pred'
        self.figure_posterior_mean = self.directory_figures + '/posterior_mean'
        self.figure_parameter_cross_section = self.directory_figures + '/parameter_cross_section'
        self.figure_posterior_covariance = self.directory_figures + '/posterior_covariance'

        #=== File Path for vtk Files ===#
        self.figure_vtk_parameter_test = self.directory_figures + '/parameter_test'
        self.figure_vtk_state_test = self.directory_figures + '/state_test'
        self.figure_vtk_parameter_pred = self.directory_figures + '/parameter_pred'
        self.figure_vtk_state_pred = self.directory_figures + '/state_pred'

        #=== File Path for Movie ===#
        self.directory_movie = '../../../../Movies/' + self.case_and_NN_name
        self.figure_posterior_mean_movie = self.directory_movie +\
                '/posterior_mean_movie'
        self.figure_parameter_cross_section_movie = self.directory_movie +\
                '/parameter_cross_section_movie'

        #=== Creating Directories ===#
        if not os.path.exists(self.directory_figures):
            os.makedirs(self.directory_figures)
