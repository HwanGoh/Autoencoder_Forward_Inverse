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
        #=== Key Strings ===#
        project_name = 'advection_diffusion_2d_'
        if options.flow_navier_stokes == True:
            flow_string = 'navier_stokes'
        if options.flow_darcy == True:
            flow_string = 'darcy'
        if options.time_stepping_erk4 == True:
            time_stepping_string = 'erk4'
        if options.time_stepping_lserk4 == True:
            time_stepping_string = 'lserk4'
        if options.time_stepping_implicit == True:
            time_stepping_string = 'imp'
        num_nodes_string = 'n%d'%(options.num_nodes)
        data_options = num_nodes_string + '_' +\
                       flow_string + '_' +\
                       time_stepping_string
        self.directory_dataset = '../../../datasets/fenics/advection_diffusion_2d/' +\
            num_nodes_string + '/' + flow_string + '_' + time_stepping_string + '/'

        #=== Data Type ===#
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
        prior_string = prior_string_blp('blp',
                                        options.prior_mean,
                                        options.prior_gamma,
                                        options.prior_delta)

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
                project_name +\
                'parameter_train_' +\
                'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
        self.input_test = directory_dataset +\
                project_name +\
                'parameter_test_' +\
                'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        if options.obs_type == 'full':
            self.output_train = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_train_' +\
                    'd%d_'%(options.num_data_train_load) + data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_test_' +\
                    'd%d_'%(options.num_data_test_load) + data_options + '_' + prior_string_test
        if options.obs_type == 'obs':
            self.output_train = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_train_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_train_load) +\
                    data_options + '_' + prior_string_train
            self.output_test = directory_dataset +\
                    project_name +\
                    'state_' + options.obs_type + '_test_' +\
                    'o%d_d%d_' %(options.num_obs_points, options.num_data_test_load) +\
                    data_options + '_' + prior_string_test

        #############
        #   Prior   #
        #############
        #=== Prior ===#
        self.prior_mean = self.directory_dataset +\
                'prior_mean_' + num_nodes_string + '_' + prior_string_train
        self.prior_covariance = self.directory_dataset +\
                'prior_covariance_' + num_nodes_string + '_' + prior_string_train
        self.prior_covariance_cholesky = self.directory_dataset +\
                'prior_covariance_cholesky_' + num_nodes_string + '_' + prior_string_train
        self.prior_covariance_cholesky_inverse = self.directory_dataset +\
                'prior_covariance_cholesky_inverse_' + num_nodes_string + '_' + prior_string_train

        ###################
        #   FEM Objects   #
        ###################
        #=== FEM Operators ===#
        self.fem_operator_spatial = self.directory_dataset +\
                'fem_operator_spatial_' + num_nodes_string
        self.fem_operator_implicit_ts = self.directory_dataset +\
                'fem_operator_implicit_ts_' + num_nodes_string
        self.fem_operator_implicit_ts_rhs = self.directory_dataset +\
                'fem_operator_implicit_ts_rhs_' + num_nodes_string

###############################################################################
#                               Prior Strings                                 #
###############################################################################
def prior_string_blp(prior_type, mean, gamma, delta):
    mean_string = value_to_string(mean)
    gamma_string = value_to_string(gamma)
    delta_string = value_to_string(delta)

    return '%s_%s_%s_%s'%(prior_type, mean_string, gamma_string, delta_string)
