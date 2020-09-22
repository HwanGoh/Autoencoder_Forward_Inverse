#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:18:12 2020
@author: hwan
"""
import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.insert(0, os.path.realpath('..'))

import numpy as np
import pandas as pd

# Import src code
from utils_io.value_to_string import value_to_string

# Import other codes
from filepaths import FilePaths
from prior_io import save_prior
from forward_functions import exponential

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Options                                   #
###############################################################################
class Options:
    #=== Data Type ===#
    exponential = 1

    #=== Select Train or Test Set ===#
    generate_train_data = 1
    generate_test_data = 0

    #=== Data Properties ===#
    num_data = 1000
    mesh_dimensions = 10000
    parameter_dimensions = 2
    num_obs_points = 50

    #=== Full Prior ===#
    prior_type_diag = True
    prior_mean_diag = 2
    prior_cov_diag_11 = 1
    prior_cov_diag_22 = 2

    #=== Diagonal Prior ===#
    prior_type_full = False
    prior_mean_full = 2
    prior_cov_full_11 = 1
    prior_cov_full_12 = 2
    prior_cov_full_22 = 3

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    ##################
    #   Setting Up   #
    ##################
    #=== Run Options ===#
    options = Options()

    #=== File Paths ===#
    filepaths = FilePaths(options)

    ###############################
    #   Generate and Save Prior   #
    ###############################
    #=== Mesh ===#
    domain = np.linspace(0, 1, options.mesh_dimensions, endpoint = True)
    np.random.seed(4)
    obs_indices = np.sort(
            np.random.choice(range(0, options.mesh_dimensions),
                options.num_obs_points, replace = False))

    #=== Prior ===#
    if options.prior_type_diag == 1:
        prior_mean = options.prior_mean_diag*np.ones(2)
        prior_covariance = np.array(
                [[options.prior_cov_diag_11, 0],[0, options.prior_cov_diag_22]])
    if options.prior_type_full == 1:
        prior_mean = options.prior_mean_full*np.ones(2)
        prior_covariance = np.array(
                [[options.prior_cov_full_11, options.prior_cov_full_12],
            [options.prior_cov_full_12, options.prior_cov_full_22]])
    prior_covariance_cholesky = np.linalg.cholesky(prior_covariance)
    prior_covariance_cholesky_inverse = np.linalg.inv(prior_covariance_cholesky)

    #=== Save prior ===#
    save_prior(filepaths, prior_mean, prior_covariance,
            prior_covariance_cholesky, prior_covariance_cholesky_inverse)

    ###############################################################################
    #                               Generate Data                                 #
    ###############################################################################
    parameter = np.zeros((options.num_data, options.parameter_dimensions))
    state = np.zeros((options.num_data, options.num_obs_points))

    for n in range(0, options.num_data):
        epsilon = np.random.normal(0, 1, options.parameter_dimensions)
        parameter[n,:] = np.matmul(prior_covariance_cholesky, epsilon) + prior_mean
        if exponential == 1:
            state[n,:], _ = exponential(parameter[n,:], options.obs_points,
                                        parameter_dimensions, options.num_obs_points)

    print('Data Generated')

    ###############################################################################
    #                          Saving Data as Dataframe                           #
    ###############################################################################
