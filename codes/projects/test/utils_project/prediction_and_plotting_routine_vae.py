#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import sys
sys.path.append('../../../../..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import scipy.stats as st

# Import src code
from utils_data.data_handler import DataHandler
from neural_networks.nn_vae import VAE
from utils_misc.positivity_constraints import positivity_constraint_log_exp

# Import project utilities
from utils_project.solve_forward_1d import SolveForward1D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, options, filepaths):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.mesh_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(filepaths.project.obs_indices + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, filepaths,
                       options.parameter_dimensions, obs_dimensions)
    data.load_data_test()
    if options.add_noise == 1:
        data.add_noise_output_test()
    parameter_test = data.input_test
    state_obs_test = data.output_test

    #=== Load Trained Neural Network ===#
    NN = VAE(hyperp, options,
             input_dimensions, latent_dimensions,
             None, None,
             positivity_constraint_log_exp)
    NN.load_weights(filepaths.trained_NN)

    #=== Construct Forward Model ===#
    if options.model_augmented == True:
        forward_model = SolveForward1D(options, filepaths, obs_indices)
        if options.exponential == True:
            forward_model_solve = forward_model.exponential

    #=== Selecting Samples ===#
    sample_number = 105
    parameter_test_sample = np.expand_dims(parameter_test[sample_number,:], 0)
    state_obs_test_sample = np.expand_dims(state_obs_test[sample_number,:], 0)

    #=== Predictions ===#
    posterior_mean_pred, posterior_cov_pred = NN.encoder(state_obs_test_sample)
    n_samples = 1000
    posterior_pred_draws = np.zeros((n_samples, posterior_mean_pred.shape[1]))
    state_obs_pred_draws = np.zeros((n_samples, state_obs_test_sample.shape[1]))
    for n in range(0,n_samples):
        posterior_pred_draws[n,:] = NN.reparameterize(posterior_mean_pred, posterior_cov_pred)
    if options.model_aware == True:
        state_obs_pred_draws = NN.decoder(posterior_pred_draws)
    else:
        state_obs_pred_draws = forward_model_solve(posterior_pred_draws)

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')
    n_bins = 100
    for n in range(0, posterior_mean_pred.shape[1]):
        #=== Posterior Histogram ===#
        plt.hist(posterior_pred_draws[:,n], density=True,
                 range=[-1,4], bins=n_bins)
        #=== True Parameter Value ===#
        plt.axvline(parameter_test_sample[0,n], color='r',
                linestyle='dashed', linewidth=3,
                label="True Parameter Value")
        #=== Predicted Posterior Mean ===#
        plt.axvline(posterior_mean_pred[0,n], color='b',
                linestyle='dashed', linewidth=1,
                label="Predicted Posterior Mean")
        #=== Probability Density Function ===#
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 301)
        kde = st.gaussian_kde(posterior_pred_draws[:,n])
        #=== Title and Labels ===#
        plt.plot(kde_xs, kde.pdf(kde_xs))
        plt.legend(loc="upper left")
        plt.ylabel('Probability')
        plt.xlabel('Parameter Value')
        plt.title("Marginal Posterior Parameter_%d"%(n));
        #=== Save and Close Figure ===#
        plt.savefig(filepaths.figure_parameter_pred + '_%d'%(n))
        plt.close()

    print('Predictions plotted')

###############################################################################
#                                Plot Metrics                                 #
###############################################################################
def plot_and_save_metrics(hyperp, options, filepaths):
    print('================================')
    print('        Plotting Metrics        ')
    print('================================')
    #=== Load Metrics ===#
    print('Loading Metrics')
    df_metrics = pd.read_csv(filepaths.trained_NN + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()

    ####################
    #   Load Metrics   #
    ####################
    storage_array_loss_train = array_metrics[:,0]
    storage_array_loss_train_VAE = array_metrics[:,1]
    storage_array_loss_train_encoder = array_metrics[:,2]
    storage_array_relative_error_input_VAE = array_metrics[:,10]
    storage_array_relative_error_latent_post_draw = array_metrics[:,11]
    storage_array_relative_error_input_decoder = array_metrics[:,12]
    storage_array_relative_gradient_norm = array_metrics[:,13]

    ################
    #   Plotting   #
    ################
    #=== Loss Train ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train))
    plt.title('Log-Loss for Training Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Autoencoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_VAE))
    plt.title('Log-Loss for VAE')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss_autoencoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Loss Encoder ===#
    fig_loss = plt.figure()
    x_axis = np.linspace(1, hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, np.log(storage_array_loss_train_encoder))
    plt.title('Log-Loss for Encoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'loss_encoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    #=== Relative Error Autoencoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_VAE)
    plt.title('Relative Error for Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_autoencoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Error Posterior Draw ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_latent_post_draw)
    plt.title('Relative Error for Posterior Draw')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_post_draw.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Error Decoder ===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_error_input_decoder)
    plt.title('Relative Error for Decoder')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_decoder.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    #=== Relative Gradient Norm ===#
    fig_gradient_norm = plt.figure()
    x_axis = np.linspace(1,hyperp.num_epochs, hyperp.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_relative_gradient_norm)
    plt.title('Relative Gradient Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = filepaths.directory_figures + '/' +\
            'relative_error_gradient_norm.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_gradient_norm)

    print('Plotting complete')
