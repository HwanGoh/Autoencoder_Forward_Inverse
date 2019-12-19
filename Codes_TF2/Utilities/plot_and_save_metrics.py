#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:09:59 2019

@author: hwan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_and_save_metrics(hyperp, run_options, file_paths, fig_size):
###############################################################################
#                               Plotting Metrics                              #
###############################################################################      
    df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()
    x_axis = np.linspace(1, hyperp.num_epochs-1, hyperp.num_epochs-1, endpoint = True)

    ######################
    #  Autoencoder Loss  #                          
    ######################
    fig = plt.figure()
    print('Loading Metrics')
    storage_loss_array = array_metrics[1:,0]
    plt.plot(x_axis, np.log(storage_loss_array), label = 'Log-Loss')
        
    #=== Figure Properties ===#   
    plt.title('Training Log-Loss of Autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_autoencoder_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    ####################
    #  Parameter Loss  #                          
    ####################
    fig = plt.figure()
    print('Loading Metrics')
    storage_parameter_loss_array = array_metrics[1:,1]
    plt.plot(x_axis, np.log(storage_parameter_loss_array), label = 'Log-Loss')
        
    #=== Figure Properties ===#   
    plt.title('Training Log-Loss of Parameter Data')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_parameter_data_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
    
    ################
    #  State Loss  #                          
    ################
    fig = plt.figure()
    print('Loading Metrics')
    storage_state_loss_array = array_metrics[1:,2]
    plt.plot(x_axis, np.log(storage_state_loss_array), label = 'Log-Loss')
        
    #=== Figure Properties ===#   
    plt.title('Training Log-Loss of State Data')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_state_data_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
    
    ##############################
    #  Parameter Relative Error  #                          
    ##############################
    fig = plt.figure()
    print('Loading Metrics')
    storage_parameter_relative_error = array_metrics[1:,7]
    plt.plot(x_axis, storage_parameter_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of Parameter Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'relative_error' + '_parameter_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
    
    ##########################
    #  State Relative Error  #                          
    ##########################
    fig = plt.figure()
    print('Loading Metrics')
    storage_state_relative_error = array_metrics[1:,8]
    plt.plot(x_axis, storage_state_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of State Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'relative_error' + '_state_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)