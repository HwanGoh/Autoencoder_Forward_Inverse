#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
import os

sys.path.insert(0, os.path.realpath('../../../../../FEniCS_Codes/src'))

import numpy as np
import pandas as pd

# Import src code
from utils_data.data_handler import DataHandler
from neural_networks.nn_vae_fwd_inv import VAEFwdInv
from utils_misc.positivity_constraints import positivity_constraint_log_exp

# Import project utilities
from utils_project.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d

# Import FEniCS code
from utils_mesh.construct_mesh_rectangular import construct_mesh

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, filepaths):

    #=== Mesh Properties ===#
    options.mesh_point_1 = [-1,-1]
    options.mesh_point_2 = [1,1]
    options.num_nodes_x = 15
    options.num_nodes_y = 15
    options.num_obs_points = 58
    options.order_fe_space = 1
    options.order_meta_space = 1
    options.num_nodes = (options.num_nodes_x + 1) * (options.num_nodes_y + 1)

    #=== Construct Mesh ===#
    fe_space, meta_space,\
    nodes, dof_fe, dof_meta = construct_mesh(options)

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, filepaths,
                       options.parameter_dimensions, obs_dimensions)
    data.load_data_test()
    parameter_test = data.input_test
    state_obs_test = data.output_test

    #=== Load Trained Neural Network ===#
    NN = VAEFwdInv(hyperp, options,
                    input_dimensions, latent_dimensions,
                    None, None,
                    positivity_constraint_log_exp)
    NN.load_weights(filepaths.trained_NN)

    #=== Selecting Samples ===#
    sample_number = 1
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    posterior_mean_pred, posterior_cov_pred = NN.encoder(state_obs_test_sample)
    posterior_pred_draw = NN.reparameterize(posterior_mean_pred, posterior_cov_pred)
    posterior_mean_pred = posterior_mean_pred.numpy().flatten()
    posterior_pred_draw = posterior_pred_draw.numpy().flatten()

    if options.model_aware == 1:
        state_obs_pred_draw = NN.decoder(np.expand_dims(posterior_pred_draw, 0))
        state_obs_pred_draw = state_obs_pred_draw.numpy().flatten()

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')

    #=== Plot FEM Functions ===#
    plot_fem_function_fenics_2d(meta_space, parameter_test_sample,
                                'True Parameter',
                                filepaths.figure_parameter_test + '.png',
                                (5,5), (0,6))
    plot_fem_function_fenics_2d(meta_space, posterior_mean_pred,
                                'Posterior Mean',
                                filepaths.figure_posterior_mean + '.png',
                                (5,5), (0,6))
    plot_fem_function_fenics_2d(meta_space, posterior_pred_draw,
                                'Posterior Draw',
                                filepaths.figure_parameter_pred + '.png',
                                (5,5), (0,6))
    if options.obs_type == 'full':
        plot_fem_function_fenics_2d(meta_space, state_obs_test_sample,
                                    'True State',
                                    filepaths.figure_state_test + '.png',
                                    (5,5))
        plot_fem_function_fenics_2d(meta_space, state_obs_pred_draw,
                                    'State Prediction',
                                    filepaths.figure_state_pred + '.png',
                                    (5,5))

    print('Predictions plotted')
