#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""

from utils_io.value_to_string import value_to_string

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePathsProject:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Defining Filenames ===#
        data_options = 'n%d'%(options.parameter_dimensions)
        directory_dataset = '../../../../../datasets/simple_1d/'
        if options.exponential == 1:
            project_name = 'exponential_1d'
        directory_dataset += project_name + '/' + data_options + '/'

        #=== Data Properties ===#
        if options.obs_type == 'full':
            obs_string = 'full'
        if options.obs_type == 'obs':
            obs_string = 'obs_o%d'%(options.num_obs_points)
        if options.add_noise == 1:
            noise_level_string = value_to_string(options.noise_level)
            noise_string = 'ns%s_%d'%(noise_level_string,options.num_noisy_obs)
        else:
            noise_string = 'ns0'
        data_string = data_options + '_' + obs_string + '_' + noise_string + '_'

        #=== Prior Properties ===#
        if options.prior_type_diag_train == True:
            prior_string_train = self.prior_string_diag('diag',
                    options.prior_mean_diag_train,
                    options.prior_cov_diag_11_train,
                    options.prior_cov_diag_22_train)
        if options.prior_type_diag_test == True:
            prior_string_test = self.prior_string_diag('diag',
                    options.prior_mean_diag_test,
                    options.prior_cov_diag_11_test,
                    options.prior_cov_diag_22_test)

        if options.prior_type_full_train == True:
            prior_string = self.prior_string_full('full',
                    options.prior_mean_full_train,
                    options.prior_cov_full_11_train,
                    options.prior_cov_full_12_train,
                    options.prior_cov_full_22_train)

        if options.prior_type_full_test == True:
            prior_string = self.prior_string_full('full',
                    options.prior_mean_full_test,
                    options.prior_cov_full_11_test,
                    options.prior_cov_full_12_test,
                    options.prior_cov_full_22_test)

        #=== Case String ===#
        self.case_name = project_name + '_' + data_string + prior_string_train

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.obs_indices = directory_dataset +\
                project_name + '_' + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.input_train = directory_dataset +\
                project_name + '_' + 'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
        self.input_test = directory_dataset +\
                project_name + '_' + 'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        #=== State ===#
        if options.obs_type == 'full':
            self.output_train = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_train_' +\
                    'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_test_' +\
                    'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        if options.obs_type == 'obs':
            self.output_train = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_train_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name + '_' + 'state_' + options.obs_type + '_test_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_test_load) +\
                    data_options + '_' + prior_string_test

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_mean = directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string_train
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string_train
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string_train

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string_diag(self, prior_type, mean, cov_11, cov_22):
        mean_string = value_to_string(mean)
        cov_11_string = value_to_string(cov_11)
        cov_22_string = value_to_string(cov_22)

        return '%s_%s_%s_%s'%(prior_type, mean_string, cov_11_string, cov_22_string)

    def prior_string_full(self, prior_type, mean, variance, corr):
        mean_string = value_to_string(mean)
        cov_11_string = value_to_string(cov_11)
        cov_12_string = value_to_string(cov_12)
        cov_22_string = value_to_string(cov_22)

        return '%s_%s_%s_%s'%(prior_type, mean_string,
                              cov_11_string, cov_12_string, cov_22_string)
