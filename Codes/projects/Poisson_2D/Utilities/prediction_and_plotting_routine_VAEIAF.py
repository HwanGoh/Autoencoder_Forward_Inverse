#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
sys.path.append('../../../..')

import numpy as np
import pandas as pd

from get_train_and_test_data import load_train_and_test_data
from NN_VAEIAF_Fwd_Inv import VAEIAFFwdInv
from positivity_constraints import positivity_constraint_log_exp
from Finite_Element_Method.src.load_mesh import load_mesh
from Utilities.plot_FEM_function import plot_FEM_function
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, file_paths,
                     project_name, data_options, dataset_directory):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Load Trained Neural Network ===#
    NN = VAEIAFFwdInv(hyperp, options,
                        input_dimensions, latent_dimensions,
                        None, None,
                        None, None,
                        positivity_constraint_log_exp)
    NN.load_weights(file_paths.NN_savefile_name)

    #=== Load Data ===#
    _, _,\
    parameter_test, state_obs_test\
    = load_train_and_test_data(file_paths,
            hyperp.num_data_train, options.num_data_test,
            options.parameter_dimensions, obs_dimensions,
            load_data_train_flag = 0,
            normalize_input_flag = 0, normalize_output_flag = 0)

    #=== Add Noise to Data ===#
    # if options.add_noise == 1:
    #     _, state_obs_test, noise_regularization_matrix\
    #     = add_noise(options, _, state_obs_test, load_data_train_flag = 0)
    # else:
    #     noise_regularization_matrix = tf.eye(obs_dimensions)

    #=== Selecting Samples ===#
    sample_number = 105
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    parameter_pred_sample, _ = NN.IAF_chain_posterior(
            NN.encoder(state_obs_test_sample))
    state_obs_pred_sample = NN.decoder(parameter_test_sample)
    parameter_pred_sample = parameter_pred_sample.numpy().flatten()
    state_obs_pred_sample = state_obs_pred_sample.numpy().flatten()

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')
    #=== Load Mesh ===#
    nodes, elements, _, _, _, _, _, _ = load_mesh(file_paths)

    #=== Plot FEM Functions ===#
    plot_FEM_function(file_paths.figures_savefile_name_parameter_test,
                     'True Parameter', 7.0,
                      nodes, elements,
                      parameter_test_sample)
    plot_FEM_function(file_paths.figures_savefile_name_parameter_pred,
                      'Parameter Prediction', 7.0,
                      nodes, elements,
                      parameter_pred_sample)
    if options.obs_type == 'full':
        plot_FEM_function(file_paths.figures_savefile_name_state_test,
                          'True State', 2.6,
                          nodes, elements,
                          state_obs_test_sample)
        plot_FEM_function(file_paths.figures_savefile_name_state_pred,
                          'State Prediction', 2.6,
                          nodes, elements,
                          state_obs_pred_sample)

    print('Predictions plotted')

###############################################################################
#                                Plot Metrics                                 #
###############################################################################
def plot_and_save_metrics(hyper_p, options, file_paths):
    print('================================')
    print('        Plotting Metrics        ')
    print('================================')
    #=== Load Metrics ===#
    print('Loading Metrics')
    df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()

    ####################
    #   Load Metrics   #
    ####################
    storage_array_loss_train = array_metrics[:,0]
    storage_array_loss_train_VAE = array_metrics[:,1]
    storage_array_loss_train_encoder = array_metrics[:,2]
    storage_array_relative_error_input_VAE = array_metrics[:,10]
    storage_array_relative_error_latent_encoder = array_metrics[:,11]
    storage_array_relative_error_input_decoder = array_metrics[:,12]
    storage_array_relative_gradient_norm = array_metrics[:,13]

    ################
    #   Plotting   #
    ################
    #=== Loss Train ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train))
    plt.title('Log-Loss for Training Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'loss.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Autoencoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_VAE))
    plt.title('Log-Loss for VAE')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'loss_autoencoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Encoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_encoder))
    plt.title('Log-Loss for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'loss_encoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Relative Error Autoencoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_VAE)
    plt.title('Relative Error for Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_autoencoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Error Encoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_latent_encoder)
    plt.title('Relative Error for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_encoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Error Decoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_decoder)
    plt.title('Relative Error for Decoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_decoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Gradient Norm ===#
    fig_gradient_norm = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_gradient_norm)
    plt.title('Relative Gradient Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_gradient_norm.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_gradient_norm)

    if options.model_augmented == 1:
        #=== Relative Error Decoder ===#
        fig_loss = plt.figure()
        x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
        plt.plot(x_axis, storage_array_loss_train_forward_model)
        plt.title('Log-loss Forward Model')
        plt.xlabel('Epochs')
        plt.ylabel('Relative Error')
        figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
                'loss_forward_model.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_loss)

    print('Plotting complete')
