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

# Import data generator codes
from filepaths import FilePaths
from prior_laplace_finite_difference import prior_laplace_finite_difference
from prior_io import save_prior
from forward_functions import continuous_linear
from dataset_io import save_forward_vector, save_parameter, save_state

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Options                                   #
###############################################################################
class Options:
    #=== Data Type ===#
    continuous_linear = 1

    #=== Select Train or Test Set ===#
    generate_train_data = 1
    generate_test_data = 0

    #=== Data Properties ===#
    num_data = 5000
    mesh_dimensions = 100
    parameter_dimensions = mesh_dimensions
    num_obs_points = 1
    noise_level = 0.3

    #=== Identity Prior ===#
    prior_type_identity = True
    prior_mean_identity = 0

    #=== Laplacian Prior ===#
    prior_type_laplacian = False
    prior_mean_laplacian = 0

    #=== Random Seed ===#
    random_seed = 4

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
    mesh = np.linspace(0, 1, options.mesh_dimensions, endpoint = True)

    #=== Prior ===#
    prior_mean = options.prior_mean*np.ones(len(mesh))
    prior_covariance = np.eye(options.parameter_dimensions)
    prior_covariance_cholesky = np.linalg.cholesky(prior_covariance)
    prior_covariance_cholesky_inverse = np.linalg.inv(prior_covariance_cholesky)

    #=== Save Prior ===#
    save_prior(filepaths, prior_mean, prior_covariance,
               prior_covariance_cholesky, prior_covariance_cholesky_inverse)

    ##############################
    #   Generate and Save Data   #
    ##############################
    parameter = np.zeros((options.num_data, options.parameter_dimensions))
    state = np.zeros((options.num_data, options.mesh_dimensions))

    #=== Generate Parameters ===#
    for n in range(0, options.num_data):
        epsilon = np.random.normal(0, 1, options.parameter_dimensions)
        parameter[n,:] = np.matmul(prior_covariance_cholesky, epsilon) + prior_mean

    #=== Generate State ===#
    np.random.seed(options.random_seed)
    forward_vector = np.random.uniform(2, 10, options.parameter_dimensions)
    save_forward_vector(filepaths, forward_vector)
    for n in range(0, options.num_data):
        state[n,:] = continuous_linear(parameter[n,:], forward_vector)

    #=== Generate Observation Data ===#
    obs_indices = np.zeros(1, dtype=int)
    state_obs = state[:,obs_indices]

    #=== Save Dataset ====#
    save_parameter(filepaths, parameter)
    save_state(filepaths, obs_indices, state, state_obs)
    print('Data Generated and Saved')
