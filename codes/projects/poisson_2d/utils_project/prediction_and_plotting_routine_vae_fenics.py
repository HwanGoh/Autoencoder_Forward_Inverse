#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
import os

sys.path.insert(0, os.path.realpath('../../../../../fenics-simulations/src'))

import numpy as np
import pandas as pd

# Import src code
from utils_data.data_handler import DataHandler
from neural_networks.nn_vae import VAE
from utils_misc.positivity_constraints import positivity_constraint_log_exp

# Import project utilities
from utils_project.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_project.plot_cross_section import plot_cross_section

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
    options.num_obs_points = 10
    options.order_fe_space = 1
    options.order_meta_space = 1
    options.num_nodes = (options.num_nodes_x + 1) * (options.num_nodes_y + 1)

    # options.mesh_point_1 = [-1,-1]
    # options.mesh_point_2 = [1,1]
    # options.num_nodes_x = 30
    # options.num_nodes_y = 30
    # options.num_obs_points = 118
    # options.order_fe_space = 1
    # options.order_meta_space = 1
    # options.num_nodes = (options.num_nodes_x + 1) * (options.num_nodes_y + 1)

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
    if options.add_noise == True:
        data.add_noise_output_test()
    parameter_test = data.input_test
    state_obs_test = data.output_test

    ##=== Load Trained Neural Network ===#
    NN = VAE(hyperp, options,
             input_dimensions, latent_dimensions,
             None, None,
             positivity_constraint_log_exp)
    NN.load_weights(filepaths.trained_NN)

    #=== Selecting Samples ===#
    sample_number = 2
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Saving Specific Sample ===#
    df_input_specific = pd.DataFrame({'input_specific': parameter_test_sample.flatten()})
    df_input_specific.to_csv(filepaths.input_specific + '.csv', index=False)
    df_output_specific = pd.DataFrame({'output_specific': state_obs_test_sample.flatten()})
    df_output_specific.to_csv(filepaths.output_specific + '.csv', index=False)

    #=== Predictions ===#
    posterior_mean_pred, posterior_cov_pred = NN.encoder(state_obs_test_sample)
    posterior_pred_draw = NN.reparameterize(posterior_mean_pred, posterior_cov_pred)

    posterior_mean_pred = posterior_mean_pred.numpy().flatten()
    posterior_cov_pred = posterior_cov_pred.numpy().flatten()
    posterior_pred_draw = posterior_pred_draw.numpy().flatten()

    if options.model_aware == 1:
        state_obs_pred_draw = NN.decoder(np.expand_dims(posterior_pred_draw, 0))
        state_obs_pred_draw = state_obs_pred_draw.numpy().flatten()

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')

    #=== Plot FEM Functions ===#
    cross_section_y = 0.5
    plot_parameter_min = 1.5
    plot_parameter_max = 8
    plot_variance_min = 0
    plot_variance_max = 1.
    filename_extension = '_%d.png'%(sample_number)
    plot_fem_function_fenics_2d(meta_space, parameter_test_sample,
                                cross_section_y,
                                '',
                                filepaths.figure_parameter_test + filename_extension,
                                (5,5), (plot_parameter_min,plot_parameter_max),
                                True)
    plot_fem_function_fenics_2d(meta_space, posterior_mean_pred,
                                cross_section_y,
                                '',
                                filepaths.figure_posterior_mean + filename_extension,
                                (5,5), (plot_parameter_min,plot_parameter_max),
                                False)
    plot_fem_function_fenics_2d(meta_space, posterior_pred_draw,
                                cross_section_y,
                                '',
                                filepaths.figure_parameter_pred + filename_extension,
                                (5,5), (plot_parameter_min,plot_parameter_max),
                                False)
    if options.obs_type == 'full':
        plot_fem_function_fenics_2d(meta_space, state_obs_test_sample,
                                    cross_section_y,
                                    'True State',
                                    filepaths.figure_state_test + filename_extension,
                                    (5,5))
        plot_fem_function_fenics_2d(meta_space, state_obs_pred_draw,
                                    cross_section_y,
                                    'State Prediction',
                                    filepaths.figure_state_pred + filename_extension,
                                    (5,5))

    #=== Plot Cross-Section with Error Bounds ===#
    plot_cross_section(meta_space,
                       parameter_test_sample, posterior_mean_pred, posterior_cov_pred,
                       (-1,1), cross_section_y,
                       '',
                       filepaths.figure_parameter_cross_section + filename_extension,
                       (plot_parameter_min,plot_parameter_max))

    #=== Plot Variation ===#
    plot_fem_function_fenics_2d(meta_space, np.exp(posterior_cov_pred),
                                cross_section_y,
                                '',
                                filepaths.figure_posterior_covariance + filename_extension,
                                (5,5), (plot_variance_min,plot_variance_max),
                                False)

    print('Predictions plotted')
