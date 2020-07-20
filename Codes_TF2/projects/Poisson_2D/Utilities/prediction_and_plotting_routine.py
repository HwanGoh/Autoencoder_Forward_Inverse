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
from NN_AE import AutoencoderFwdInv
from Finite_Element_Method.src.load_mesh import load_mesh
from Utilities.plot_FEM_function import plot_FEM_function
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory):

    #=== Load Observation Indices ===#
    if run_options.obs_type == 'full':
        obs_dimensions = run_options.parameter_dimensions
        obs_indices = []
    if run_options.obs_type == 'obs':
        obs_dimensions = run_options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Data and Latent Dimensions of Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        input_dimensions = run_options.parameter_dimensions
        latent_dimensions = obs_dimensions
    if run_options.use_reverse_autoencoder == 1:
        input_dimensions = obs_dimensions
        latent_dimensions = run_options.parameter_dimensions

    #=== Load Trained Neural Network ===#
    NN = AutoencoderFwdInv(hyperp,
            input_dimensions, latent_dimensions,
            None, None)
    NN.load_weights(file_paths.NN_savefile_name)

    #=== Loading Data ===#
    _, _,\
    parameter_test, state_obs_test,\
    = load_train_and_test_data(file_paths,
            run_options.num_data_train, run_options.num_data_test,
            run_options.parameter_dimensions, obs_dimensions,
            load_data_train_flag = 0,
            normalize_input_flag = 0, normalize_output_flag = 0)

    #=== Selecting Samples ===#
    sample_number = 120
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    if run_options.use_standard_autoencoder == 1:
        state_obs_pred_sample = NN.encoder(parameter_test_sample)
        parameter_pred_sample = NN.decoder(state_obs_test_sample)
    if run_options.use_reverse_autoencoder == 1:
        state_obs_pred_sample = NN.decoder(parameter_test_sample)
        parameter_pred_sample = NN.encoder(state_obs_test_sample)
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
                     'True Parameter', 5.0,
                      nodes, elements,
                      parameter_test_sample)
    plot_FEM_function(file_paths.figures_savefile_name_parameter_pred,
                      'Parameter Prediction', 5.0,
                      nodes, elements,
                      parameter_pred_sample)
    if run_options.obs_type == 'full':
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
def plot_and_save_metrics(hyper_p, run_options, file_paths):
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
    storage_array_loss_train_autoencoder = array_metrics[:,1]
    storage_array_loss_train_encoder = array_metrics[:,2]
    storage_array_loss_train_decoder = array_metrics[:,3]
    if run_options.use_model_aware == 1:
        #=== Metrics ===#
        storage_array_relative_error_input_autoencoder = array_metrics[:,8]
        storage_array_relative_error_input_encoder = array_metrics[:,9]
        storage_array_relative_error_input_decoder = array_metrics[:,10]
        storage_array_relative_gradient_norm = array_metrics[:,11]
    if run_options.use_model_augmented == 1:
        #=== Metrics ===#
        storage_array_loss_train_forward_model = array_metrics[:,4]
        storage_array_relative_error_input_autoencoder = array_metrics[:,10]
        storage_array_relative_error_input_encoder = array_metrics[:,11]
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
            'loss' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Autoencoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_autoencoder))
    plt.title('Log-Loss for Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'loss_autoencoder' + '_' + file_paths.filename + '.png'
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
            'loss_encoder' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Decoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_decoder))
    plt.title('Log-Loss for Decoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'loss_decoder' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Relative Error Autoencoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_autoencoder)
    plt.title('Relative Error for Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_autoencoder' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Error Encoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_encoder)
    plt.title('Relative Error for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'relative_error_encoder' + '_' + file_paths.filename + '.png'
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
            'relative_error_decoder' + '_' + file_paths.filename + '.png'
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
            'relative_error_gradient_norm' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_gradient_norm)

    if run_options.use_model_augmented == 1:
        #=== Relative Error Decoder ===#
        fig_loss = plt.figure()
        x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
        plt.plot(x_axis, storage_array_loss_train_forward_model)
        plt.title('Log-loss Forward Model')
        plt.xlabel('Epochs')
        plt.ylabel('Relative Error')
        figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
                'loss_forward_model' + '_' + file_paths.filename + '.png'
        plt.savefig(figures_savefile_name)
        plt.close(fig_loss)

    print('Plotting complete')
