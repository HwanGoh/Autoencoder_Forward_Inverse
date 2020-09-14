#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:09:59 2019

@author: hwan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_and_save_metrics(hyperp, run_options, file_paths, fig_size):
###############################################################################
#                               Plotting Metrics                              #
###############################################################################      
    df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()
    x_axis = np.linspace(1, hyperp.num_epochs-1, hyperp.num_epochs-1, endpoint = True)

    ###############
    #  Full Loss  #                          
    ###############
    fig = plt.figure()
    print('Loading Metrics')
    storage_loss_array = array_metrics[1:,0]
    plt.plot(x_axis, np.log(storage_loss_array), label = 'Log-Loss')
        
    #=== Figure Properties ===#   
    plt.title('Training Log-Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log-Loss')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_full_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    ######################
    #  Autoencoder Loss  #                          
    ######################
    fig = plt.figure()
    print('Loading Metrics')
    storage_autoncoder_loss_array = array_metrics[1:,1]
    plt.plot(x_axis, np.log(storage_autoncoder_loss_array), label = 'Log-Loss')
        
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
    
    ##################
    #  Encoder Loss  #                          
    ##################
    if run_options.use_model_aware == 1 or run_options.use_model_induced == 1:
        fig = plt.figure()
        print('Loading Metrics')
        storage_encoder_loss_array = array_metrics[1:,2]
        plt.plot(x_axis, np.log(storage_encoder_loss_array), label = 'Log-Loss')
            
        #=== Figure Properties ===#   
        plt.title('Training Log-Loss of Encoder')
        plt.xlabel('Epochs')
        plt.ylabel('Log-Loss')
        #plt.axis([0,30,1.5,3])
        plt.legend()
        
        #=== Saving Figure ===#
        figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_encoder_' + file_paths.filename + '.png'
        plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)
    
    ##################
    #  Augment Loss  #                          
    ##################
    if run_options.use_model_augmented == 1 or run_options.use_model_induced == 1:
        fig = plt.figure()
        print('Loading Metrics')
        if run_options.use_model_augmented == 1:
            storage_augment_loss_array = array_metrics[1:,2]
        if run_options.use_model_induced == 1:
            storage_augment_loss_array = array_metrics[1:,3]
        plt.plot(x_axis, np.log(storage_augment_loss_array), label = 'Log-Loss')
            
        #=== Figure Properties ===#   
        plt.title('Training Log-Loss of Model Augment')
        plt.xlabel('Epochs')
        plt.ylabel('Log-Loss')
        #plt.axis([0,30,1.5,3])
        plt.legend()
        
        #=== Saving Figure ===#
        figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'loss' + '_augment_' + file_paths.filename + '.png'
        plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)
    
    ############################
    #  Encoder Relative Error  #                          
    ############################
    fig = plt.figure()
    print('Loading Metrics')
    if run_options.use_model_induced == 1:
        storage_encoder_relative_error = array_metrics[1:,9]
    else:
        storage_encoder_relative_error = array_metrics[1:,7]
    plt.plot(x_axis, storage_encoder_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of Encoder Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'relative_error' + '_encoder_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
    
    ############################
    #  Decoder Relative Error  #                          
    ############################
    fig = plt.figure()
    print('Loading Metrics')
    if run_options.use_model_induced == 1:
        storage_decoder_relative_error = array_metrics[1:,10]
    else:
        storage_decoder_relative_error = array_metrics[1:,8]
    plt.plot(x_axis, storage_decoder_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of Decoder Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = file_paths.figures_savefile_directory + '/' + 'relative_error' + '_decoder_' + file_paths.filename + '.png'
    plt.savefig(figures_savefile_name, bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)