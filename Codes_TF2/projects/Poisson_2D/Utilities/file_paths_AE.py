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
#                               Value to String                               #
###############################################################################
def value_to_string(value):
    if value >= 1:
        value = int(value)
        string = str(value)
    else:
        value_decimal_form = '%.9f'%(value)
        string = 'pt'
        nonzero_value_flag = 0
        for n in range(2,9):
            if value_decimal_form[n] == '0':
                string += '0'
            else:
                string += value_decimal_form[n]
                nonzero_value_flag = 1
                break
        if nonzero_value_flag != 1:
            string = '0'

    return string

###############################################################################
#                               Prior Strings                                 #
###############################################################################
def prior_string_AC(prior_type, mean, variance, corr):
    mean_string = value_to_string(mean)
    variance_string = value_to_string(variance)
    corr_string = value_to_string(corr)

    return '%s_%s_%s_%s'%(prior_type, mean_string, variance_string, corr_string)

def prior_string_matern(prior_type, kern_type, cov_length):
    cov_length_string = value_to_string(cov_length)

    return '%s_%s_%s'%(prior_type, kern_type, cov_length)

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
        if run_options.add_noise == 1:
            noise_level_string = value_to_string(run_options.noise_level)
            noise_string = 'ns%s_%d'%(noise_level_string,run_options.num_noisy_obs)
        else:
            noise_string = 'ns0'
        data_string = data_options + '_' + obs_string + '_' + noise_string + '_'

        #=== Prior Properties ===#
        if run_options.prior_type_AC_train == 1:
            prior_string_train = prior_string_AC('AC',
                    run_options.prior_mean_AC_train,
                    run_options.prior_variance_AC_train,
                    run_options.prior_corr_AC_train)
        if run_options.prior_type_AC_test == 1:
            prior_string_test = prior_string_AC('AC',
                    run_options.prior_mean_AC_test,
                    run_options.prior_variance_AC_test,
                    run_options.prior_corr_AC_test)

        if run_options.prior_type_matern_train == 1:
            prior_string_train = prior_string_matern('matern',
                    run_options.prior_kern_type_train,
                    run_options.prior_cov_length_train)
        if run_options.prior_type_matern_test == 1:
            prior_string_test = prior_string_matern('matern',
                    run_options.prior_kern_type_test,
                    run_options.prior_cov_length_test)

        #=== Neural Network Architecture ===#
        if run_options.standard_autoencoder == 1:
            autoencoder_type = 'AE_std_'
        if run_options.reverse_autoencoder == 1:
            autoencoder_type = 'AE_rev_'

        #=== Penalty Strings ===#
        penalty_encoder_string = value_to_string(hyperp.penalty_encoder)
        penalty_decoder_string = value_to_string(hyperp.penalty_decoder)
        if autoencoder_loss == 'maug_':
            penalty_aug_string = value_to_string(hyperp.penalty_aug)
        penalty_prior_string = value_to_string(hyperp.penalty_prior)

        #=== File Name ===#
        if autoencoder_loss == 'maware_':
            self.filename = project_name +\
                data_string + prior_string_train + '_' +\
                autoencoder_type + autoencoder_loss +\
                'urg%d_hle%d_hld%d_hne%d_hnd%d_%s_en%s_de%s_pr%s_d%d_b%d_e%d' %(
                        run_options.num_noisy_obs_unregularized,
                        hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                        hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                        hyperp.activation, penalty_encoder_string, penalty_decoder_string,
                        penalty_prior_string,
                        hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        if autoencoder_loss == 'maug_':
            self.filename = project_name +\
                data_string + prior_string_train + '_' +\
                autoencoder_type + autoencoder_loss +\
                'urg%d_hle%d_hld%d_hne%d_hnd%d_%s_en%s_de%s_aug%s_pr%s_d%d_b%d_e%d' %(
                        run_options.num_noisy_obs_unregularized,
                        hyperp.num_hidden_layers_encoder, hyperp.num_hidden_layers_decoder,
                        hyperp.num_hidden_nodes_encoder, hyperp.num_hidden_nodes_decoder,
                        hyperp.activation, penalty_encoder_string, penalty_decoder_string,
                        penalty_aug_string,
                        penalty_prior_string,
                        hyperp.num_data_train, hyperp.batch_size, hyperp.num_epochs)

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
                'd%d_'%(run_options.num_data_train_load) + data_options + '_' + prior_string_train
        self.input_test_savefilepath = dataset_directory +\
                project_name +\
                'parameter_test_' +\
                'd%d_'%(run_options.num_data_test_load) + data_options + '_' + prior_string_test
        if run_options.obs_type == 'full':
            self.output_train_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_train_' +\
                    'd%d_'%(run_options.num_data_train_load) + data_options + '_' + prior_string_train
            self.output_test_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_test_' +\
                    'd%d_'%(run_options.num_data_test_load) + data_options + '_' + prior_string_test
        if run_options.obs_type == 'obs':
            self.output_train_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_train_' +\
                    'o%d_d%d_' %(run_options.num_obs_points, run_options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.output_test_savefilepath = dataset_directory +\
                    project_name +\
                    'state_' + run_options.obs_type + '_test_' +\
                    'o%d_d%d_' %(run_options.num_obs_points, run_options.num_data_test_load) +\
                    data_options + '_' + prior_string_test

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_mean_savefilepath = dataset_directory +\
                'prior_mean_' + data_options + '_' + prior_string_train
        self.prior_covariance_savefilepath = dataset_directory +\
                'prior_covariance_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky_savefilepath = dataset_directory +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky_inverse_savefilepath = dataset_directory +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string_train

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

        #=== Mesh ===# For plotting FEM function
        mesh_name = 'mesh_square_2D_n%d' %(run_options.parameter_dimensions)
        mesh_directory = '../../../../Datasets/Mesh/' + mesh_name + '/'
        self.mesh_nodes_savefilepath = mesh_directory + mesh_name + '_nodes.csv'
        self.mesh_elements_savefilepath = mesh_directory + mesh_name + '_elements.csv'
        self.mesh_boundary_indices_edges_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices_edges.csv'
        self.mesh_boundary_indices_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices.csv'
        self.mesh_boundary_indices_bottom_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices_bottom.csv'
        self.mesh_boundary_indices_left_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices_left.csv'
        self.mesh_boundary_indices_right_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices_right.csv'
        self.mesh_boundary_indices_top_savefilepath = mesh_directory + mesh_name +\
                '_boundary_indices_top.csv'

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
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test'
        self.savefile_name_state_test = self.NN_savefile_directory + '/state_test'

        #=== File Path for Loading Displayable Predictions ===#
        self.savefile_name_parameter_pred = self.NN_savefile_directory + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_directory + '_state_pred'

        #=== File Path for Saving Figures ===#
        self.figures_savefile_directory = '../../../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
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
