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
    def __init__(self, hyperp, run_options, autoencoder_loss, dataset_directory):
        #################
        #   File Name   #
        #################
        #=== Data Type ===#
        if run_options.data_thermal_fin_nine == 1:
            dataset = 'thermalfin9'
            parameter_type = '_nine'
        if run_options.data_thermal_fin_vary == 1:
            dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if run_options.fin_dimensions_2D == 1:
            fin_dimension = ''
        if run_options.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        N_Nodes = '_' + str(run_options.full_domain_dimensions) # Must begin with an underscore!
        if run_options.fin_dimensions_2D == 1 and run_options.full_domain_dimensions == 1446:
            N_Nodes = ''
        if run_options.fin_dimensions_3D == 1 and run_options.full_domain_dimensions == 4090:
            N_Nodes = ''

        data_string = dataset + N_Nodes + '_' + hyperp.data_type + fin_dimension

        #=== Neural Network Architecture and Regularization ===#
        if run_options.use_standard_autoencoder == 1:
            autoencoder_type = 'std_'
        if run_options.use_reverse_autoencoder == 1:
            autoencoder_type = 'rev_'
        if hyperp.penalty_encoder >= 1:
            hyperp.penalty_encoder = int(hyperp.penalty_encoder)
            penalty_encoder_string = str(hyperp.penalty_encoder)
        else:
            penalty_encoder_string = str(hyperp.penalty_encoder)
            penalty_encoder_string = 'pt' + penalty_encoder_string[2:]
        if hyperp.penalty_decoder >= 1:
            hyperp.penalty_decoder = int(hyperp.penalty_decoder)
            penalty_decoder_string = str(hyperp.penalty_decoder)
        else:
            penalty_decoder_string = str(hyperp.penalty_decoder)
            penalty_decoder_string = 'pt' + penalty_decoder_string[2:]
        if autoencoder_loss == 'mind_':
            if hyperp.penalty_aug >= 1:
                hyperp.penalty_aug = int(hyperp.penalty_aug)
                penalty_aug_string = str(hyperp.penalty_aug)
            else:
                penalty_aug_string = str(hyperp.penalty_aug)
                penalty_aug_string = 'pt' + penalty_aug_string[2:]
        if hyperp.penalty_prior >= 1:
            hyperp.penalty_prior = int(hyperp.penalty_prior)
            penalty_prior_string = str(hyperp.penalty_prior)
        else:
            penalty_prior_string = str(hyperp.penalty_prior)
            penalty_prior_string = 'pt' + penalty_prior_string[2:]

        #=== File Name ===#
        if autoencoder_loss == 'maware_':
            self.filename = autoencoder_type + autoencoder_loss +\
                    data_string +\
                    '_hl%d_tl%d_hn%d_%s_en%s_de%s_pr%s_d%d_b%d_e%d' %(
                            hyperp.num_hidden_layers, hyperp.truncation_layer,
                            hyperp.num_hidden_nodes, hyperp.activation,
                            penalty_encoder_string, penalty_decoder_string,
                            penalty_prior_string,
                            run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        if autoencoder_loss == 'maug_':
            self.filename = autoencoder_type + autoencoder_loss +\
                    data_string +\
                    '_hl%d_tl%d_hn%d_%s_en%s_de%s_aug%s_pr%s_d%d_b%d_e%d' %(
                            hyperp.num_hidden_layers, hyperp.truncation_layer,
                            hyperp.num_hidden_nodes, hyperp.activation,
                            penalty_encoder_string, penalty_decoder_string,
                            penalty_aug_string,
                            penalty_prior_string,
                            run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        ################
        #   Datasets   #
        ################
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath =\
                dataset_directory +\
                'obs_indices_' + hyperp.data_type + fin_dimension
        self.input_train_savefilepath =\
                dataset_directory +\
                'parameter_train_%d'%(run_options.num_data_train) +\
                fin_dimension + parameter_type
        self.output_train_savefilepath =\
                dataset_directory +\
                'state_train_%d'%(run_options.num_data_train) +\
                fin_dimension + '_' + hyperp.data_type + parameter_type
        self.input_test_savefilepath =\
                dataset_directory +\
                'parameter_test_%d'%(run_options.num_data_test) +\
                fin_dimension +  parameter_type
        self.output_test_savefilepath =\
                dataset_directory +\
                'state_test_%d'%(run_options.num_data_test) +\
                fin_dimension + '_' + hyperp.data_type + parameter_type

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

class FilePathsPrediction(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPrediction, self).__init__(*args, **kwargs)
        #=== File Path for Loading Trained Neural Network ===#
        self.NN_savefile_directory = '../../../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename

        #=== File Path for Loading Displayable Test Data ===#
        self.loadfile_name_parameter_test = dataset_directory +\
                '/parameter_test' + fin_dimension + parameter_type
        self.loadfile_name_state_test = dataset_directory +\
                '/state_test' + fin_dimension + parameter_type

        #=== File Path for Saving Displayable Predictions ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/' +\
                'parameter_test' + fin_dimension + parameter_type
        self.savefile_name_state_test = self.NN_savefile_directory + '/' +\
                'state_test' + fin_dimension + parameter_type
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'

class FilePathsPlotting(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPlotting, self).__init__(*args, **kwargs)
        #=== File Path for Loading Displayable Predictions ===#
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'

        #=== File Path for Saving Figures ===#
        self.figures_savefile_directory = '../../../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' +\
                'parameter_test' + fin_dimension + parameter_type
        self.figures_savefile_name_state_test = self.figures_savefile_directory+ '/' +\
                'state_test' + fin_dimension + parameter_type
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)

class FilePathsPlottingParaview(FilePaths):
    def __init__(self, *args, **kwargs):
        super(FilePathsPlottingParaview, self).__init__(*args, **kwargs)
        #=== Savefile Path for Figures ===#
        self.figures_savefile_directory =\
        '/home/hwan/Documents/Github_Codes/Basic_Neural_Networks/Figures/' + self.filename
        # self.figures_savefile_directory = '../../../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' +\
                'parameter_test' + fin_dimension + parameter_type
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' +\
                'state_test' + fin_dimension + parameter_type
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
