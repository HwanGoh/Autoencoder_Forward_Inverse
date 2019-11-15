#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:01:27 2019

@author: hwan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dl

from Generate_Thermal_Fin_Data.forward_solve import Fin
from Generate_Thermal_Fin_Data.thermal_fin import get_space
from Generate_Thermal_Fin_Data.Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function

def plot_and_safe(hyper_p, run_options):
###############################################################################
#                     Form Fenics Domain and Load Predictions                 #
###############################################################################
    V,_ = get_space(40)
    solver = Fin(V) 
    
    df_parameter_test = pd.read_csv(run_options.savefile_name_parameter_test + '.csv')
    parameter_test = df_parameter_test.to_numpy()
    df_parameter_pred = pd.read_csv(run_options.savefile_name_parameter_pred + '.csv')
    parameter_pred = df_parameter_pred.to_numpy()
    
    df_state_pred = pd.read_csv(run_options.savefile_name_state_pred + '.csv')
    state_pred = df_state_pred.to_numpy()
    
###############################################################################
#                             Plotting Predictions                            #
###############################################################################
    #=== Plotting test parameter and test state ===#   
    if run_options.dataset == 'thermalfin9':
        parameter_test_dl = solver.nine_param_to_function(parameter_test)
    if run_options.dataset == 'thermalfinvary':
        parameter_test_dl = convert_array_to_dolfin_function(V,parameter_test)
        parameter_test_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_test_dl))
    if hyper_p.data_type == 'full':
        state_test_dl, _ = solver.forward(parameter_test_dl) # generate true state for comparison
        state_test = state_test_dl.vector().get_local()    
    if hyper_p.data_type == 'bndonly':
        df_state_test = pd.read_csv(run_options.savefile_name_state_test + '.csv')
        state_test = df_state_test.to_numpy()
    
    p_test_fig = dl.plot(parameter_test_dl)
    p_test_fig.ax.set_title('True Parameter', fontsize=13)  
    plt.colorbar(p_test_fig)
    plt.savefig(run_options.figures_savefile_name_parameter_test, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_parameter_test)   
    plt.show()
    
    if hyper_p.data_type == 'full': # No state prediction for bnd only data
        s_test_fig = dl.plot(state_test_dl)
        s_test_fig.ax.set_title('True State', fontsize=13) 
        plt.colorbar(s_test_fig)
        plt.savefig(run_options.figures_savefile_name_state_test, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_state_test) 
        plt.show()
    
    #=== Plotting predictions of test parameter and test state ===#
    parameter_pred_dl = solver.nine_param_to_function(parameter_pred)
    
    p_pred_fig = dl.plot(parameter_pred_dl)
    p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=13)  
    plt.colorbar(p_test_fig)
    plt.savefig(run_options.figures_savefile_name_parameter_pred, dpi=300)
    print('Figure saved to ' + run_options.figures_savefile_name_parameter_pred) 
    plt.show()
    parameter_pred_error = np.linalg.norm(parameter_pred - parameter_test,2)/np.linalg.norm(parameter_test,2)
    print(parameter_pred_error)
    
    if run_options.use_full_domain_data == 1: # No state prediction if the truncation layer only consists of the observations
        state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
        s_pred_fig = dl.plot(state_pred_dl)
        s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=13)  
        plt.colorbar(s_test_fig)
        plt.savefig(run_options.figures_savefile_name_state_pred, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_state_pred) 
        plt.show()
    state_pred_error = np.linalg.norm(state_pred - state_test,2)/np.linalg.norm(state_test,2)
    print(state_pred_error)
            
###############################################################################
#                               Plotting Metrics                              #
###############################################################################      
    df_metrics = pd.read_csv(run_options.NN_savefile_name + "_metrics" + '.csv')
    array_metrics = df_metrics.to_numpy()
    x_axis = np.linspace(1, hyper_p.num_epochs-1, hyper_p.num_epochs-1, endpoint = True)

    ######################
    #  Autoencoder Loss  #                          
    ######################
    fig_loss = plt.figure()
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
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_autoencoder_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)

    ####################
    #  Parameter Loss  #                          
    ####################
    fig_loss = plt.figure()
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
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_parameter_data_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)
    
    ################
    #  State Loss  #                          
    ################
    fig_loss = plt.figure()
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
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'loss' + '_state_data_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)
    
    ##############################
    #  Parameter Relative Error  #                          
    ##############################
    fig_loss = plt.figure()
    print('Loading Metrics')
    storage_parameter_relative_error = array_metrics[1:,7]
    plt.plot(x_axis, storage_parameter_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of State Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'relative_error' + '_parameter_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)
    
    ##########################
    #  State Relative Error  #                          
    ##########################
    fig_loss = plt.figure()
    print('Loading Metrics')
    storage_state_relative_error = array_metrics[1:,8]
    plt.plot(x_axis, storage_state_relative_error, label = 'Relative Error')
        
    #=== Figure Properties ===#   
    plt.title('Relative Error of Parameter Prediction')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error')
    #plt.axis([0,30,1.5,3])
    plt.legend()
    
    #=== Saving Figure ===#
    figures_savefile_name = run_options.figures_savefile_directory + '/' + 'relative_error' + '_state_' + run_options.filename + '.png'
    plt.savefig(figures_savefile_name)
    plt.close(fig_loss)
