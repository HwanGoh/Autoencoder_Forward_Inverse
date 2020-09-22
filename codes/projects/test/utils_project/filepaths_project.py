#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""

from utils_io.value_to_string import value_to_string

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePathsProject:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Defining Filenames ===#
        data_options = 'n%d'%(num_domain_points)
        directory_dataset = '../datasets/'
        if options.exponential == True:
            project_name = 'exponential_1d'
        directory_dataset = '../datasets/' + project_name + '/' +\
                data_options + '/'

        #=== Prior Properties ===#
        if options.prior_type_diag == True:
            prior_string = 'diag_'
            prior_string = self.prior_string_diag('diag',
                    options.prior_mean_diag,
                    options.prior_cov_diag_11,
                    options.prior_cov_diag_22)
        if options.prior_type_full == True:
            prior_string = 'full_'
            prior_string = self.prior_string_full('full',
                    options.prior_mean_diag,
                    options.prior_cov_diag_11,
                    options.prior_cov_diag_12,
                    options.prior_cov_diag_22)

        #=== Case String ===#
        self.case_name = project_name + data_string + prior_string_train

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.obs_indices = directory_dataset +\
                project_name + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.input_train = directory_dataset +\
                project_name + 'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
        self.input_test = directory_dataset +\
                project_name + 'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        #=== State ===#
        if options.obs_type == 'full':
            self.output_train = directory_dataset +\
                    project_name + 'state_' + options.obs_type + '_train_' +\
                    'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name + 'state_' + options.obs_type + '_test_' +\
                    'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        if options.obs_type == 'obs':
            self.output_train = directory_dataset +\
                    project_name + 'state_' + options.obs_type + '_train_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name + 'state_' + options.obs_type + '_test_' +\
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
