
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os

from utils_io.value_to_string import value_to_string

###############################################################################
#                              Project File Paths                             #
###############################################################################
class FilePaths:
    def __init__(self, options):

        ################
        #   Case Name  #
        ################
        #=== Defining Filenames ===#
        data_options = 'n%d'%(options.parameter_dimensions)
        directory_dataset = '../../../../../datasets/simple_1d/'
        if hasattr(options, 'continuous_linear') and options.continuous_linear == 1:
            project_name = 'continuous_linear_1d'
        if hasattr(options, 'discrete_exponential') and options.discrete_exponential == 1:
            project_name = 'discrete_exponential_1d'
        directory_dataset += project_name + '/' + data_options + '/'
        if not os.path.exists(directory_dataset):
                os.makedirs(directory_dataset)

        #=== Train of Test ===#
        if options.generate_train_data == 1:
            train_or_test = 'train_'
        if options.generate_test_data == 1:
            train_or_test = 'test_'

        #=== Prior Properties ===#
        if hasattr(options, 'prior_type_identity') and options.prior_type_identity == 1:
            prior_string = self.prior_string('identiy',
                    options.prior_mean_identity)
        if hasattr(options, 'prior_type_laplacian') and options.prior_type_laplacian == 1:
            prior_string = self.prior_string('laplacian',
                    options.prior_mean_laplacian)
        if hasattr(options, 'prior_type_diag') and options.prior_type_diag == 1:
            prior_string = self.prior_string_diag('diag',
                    options.prior_mean_diag,
                    options.prior_cov_diag_11,
                    options.prior_cov_diag_22)
        if hasattr(options, 'prior_type_full') and options.prior_type_full == 1:
            prior_string = self.prior_string_full('full',
                    options.prior_mean_full,
                    options.prior_cov_full_11,
                    options.prior_cov_full_12,
                    options.prior_cov_full_22)

        ################
        #   Datasets   #
        ################
        #=== Parameters ===#
        self.parameter = directory_dataset +\
                project_name + '_parameter_' + train_or_test +\
                'd%d_' %(options.num_data) + data_options + '_' + prior_string

        #=== State ===#
        if hasattr(options, 'continuous_linear') and options.continuous_linear == 1:
            self.forward_vector = directory_dataset +\
                    project_name + '_forward_vec_' +\
                    data_options
        self.obs_indices = directory_dataset +\
                project_name + '_obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.state_full = directory_dataset +\
                project_name + '_state_full_' + train_or_test +\
                'd%d_' %(options.num_data) +\
                data_options + '_' + prior_string
        self.state_obs = directory_dataset +\
                project_name + '_state_obs_' + train_or_test +\
                'o%d_d%d_'%(options.num_obs_points, options.num_data) +\
                data_options + '_' + prior_string

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_mean = directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string
        self.prior_covariance = directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky = directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky_inverse = directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string

###############################################################################
#                               Prior Strings                                 #
###############################################################################
    def prior_string(self, prior_type, mean):
        mean_string = value_to_string(mean)

        return '%s_%s'%(prior_type, mean_string)

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
