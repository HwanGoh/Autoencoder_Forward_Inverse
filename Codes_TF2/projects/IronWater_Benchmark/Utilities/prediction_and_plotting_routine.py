#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:53 2019

@author: hwan
"""
import numpy as np
import pandas as pd

from get_train_and_test_data import load_train_and_test_data
from NN_FC_custom import FC

import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                              Plot Predictions                               #
###############################################################################
def predict_and_plot(hyperp, run_options, file_paths,
                     project_name, data_options, dataset_directory):

    #=== Load Trained Neural Network ===#
    NN = FC(hyperp, run_options,
            run_options.parameter_dimensions, run_options.state_dimensions,
            None, None,
            None, None)
    NN.load_weights(file_paths.NN_savefile_name)

    #=== Loading Data ===#
    _, _,\
    parameter_test, state_obs_test,\
    = load_train_and_test_data(file_paths,
            run_options.num_data_train, run_options.num_data_test,
            run_options.parameter_dimensions, run_options.state_dimensions,
            load_data_train_flag = 0,
            normalize_input_flag = 0, normalize_output_flag = 0)

    state_transport_test_savefilepath =\
        dataset_directory + project_name +\
        'state_transport_test_d%d_'%(run_options.num_data_test) +\
        data_options

    state_diffusion_test_savefilepath =\
        dataset_directory + project_name +\
        'state_diffusion_test_d%d_'%(run_options.num_data_test) +\
        data_options

    df_state_transport_test = pd.read_csv(state_transport_test_savefilepath + '.csv')
    df_state_diffusion_test = pd.read_csv(state_diffusion_test_savefilepath + '.csv')
    state_transport_test = df_state_transport_test.to_numpy()
    state_diffusion_test = df_state_diffusion_test.to_numpy()
    state_transport_test = state_transport_test.reshape(
            (run_options.num_data_test, run_options.state_dimensions))
    state_diffusion_test = state_diffusion_test.reshape(
            (run_options.num_data_test, run_options.state_dimensions))

    #=== Forming Prediction ===#
    state_obs_true = state_obs_test
    state_obs_pred = NN(parameter_test)

    #=== Plotting Prediction ===#
    print('================================')
    print('      Plotting Predictions      ')
    print('================================')
    x_axis = np.linspace(1, run_options.num_data_test, run_options.num_data_test, endpoint=True)
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'prediction' + '_' + file_paths.filename + '.png'
    plt.figure(dpi=120)
    plt.plot(x_axis, state_diffusion_test*state_obs_pred,'x',label='corrected diffusion')
    plt.plot(x_axis, state_transport_test,label='transport')
    plt.plot(x_axis, state_diffusion_test,label='diffusion')
    plt.legend()
    plt.xlabel('Sample Number')
    plt.ylabel('Quantity of Interest')
    plt.title('Transport, Diffusion and Corrected Diffusion')
    plt.savefig(figures_savefile_name)

    #=== Relative Errors ===#
    relative_error_diffusion =\
            np.linalg.norm(state_transport_test - state_diffusion_test, ord=2)/\
            np.linalg.norm(state_transport_test, ord=2)
    relative_error_corrected_diffusion =\
            np.linalg.norm(state_transport_test -
                    state_diffusion_test*state_obs_pred, ord=2)/\
            np.linalg.norm(state_transport_test, ord=2)
    print('Relative Error Diffusion: %4f' %(relative_error_diffusion))
    print('Relative Error Corrected Diffusion: %4f' %(relative_error_corrected_diffusion))

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
    storage_array_loss_train = array_metrics[:,0]
    storage_array_accuracy_test = array_metrics[:,5]

    #=== Plot and Save Losses===#
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

    #=== Plot and Save Accuracies===#
    fig_accuracy = plt.figure()
    x_axis = np.linspace(1,hyper_p.num_epochs, hyper_p.num_epochs, endpoint = True)
    plt.plot(x_axis, storage_array_accuracy_test)
    plt.title('Relative Error for Modelling Discrepancy')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    figures_savefile_name = file_paths.figures_savefile_directory + '/' +\
            'accuracy' + '_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_accuracy)

    print('Plotting complete')
