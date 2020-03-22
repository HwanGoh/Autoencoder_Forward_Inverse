#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:57:30 2020

@author: hwan
"""
import sys
import os

import tensorflow as tf

from Utilities.plot_and_save_predictions_thermal_fin import plot_and_save_predictions
from Utilities.plot_and_save_predictions_vtkfiles_thermal_fin import plot_and_save_predictions_vtkfiles
from Utilities.plot_and_save_metrics import plot_and_save_metrics

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       Hyperparameters and Run_Options                       #
###############################################################################
class Hyperparameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    activation        = 'tanh'
    batch_size        = 100
    num_epochs        = 1000
    
class RunOptions:
    def __init__(self): 
        #=== Autoencoder Type ===# # Can re-use predict_and_save.py
        self.use_reverse_autoencoder = 1
        
        #=== Use Distributed Strategy ===#
        self.use_distributed_training = 0
        
        #=== Which GPUs to Use for Distributed Strategy ===#
        self.dist_which_gpus = '0,1,2'
        
        #=== Which Single GPU to Use ===#
        self.which_gpu = '3'
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_data_train = 50000
        self.num_data_test = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 1
        self.fin_dimensions_3D = 0
        
        #=== Posterior Covariance Shape ===#
        self.diagonal_posterior_covariance = 1
        self.full_posterior_covariance = 0
        
        #=== Matern and Square Exponential Prior Properties ===#
        self.prior_type_nonelliptic = 1
        self.kern_type = 'm32'
        self.prior_cov_length = 0.8
        
        #=== Elliptic Prior Properties ===#
        self.prior_type_elliptic = 0
        self.prior_type = 'elliptic'
        self.prior_elliptic_d_p = 1
        self.prior_elliptic_g_p = 0.0001
        
        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if self.data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if self.data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions

###############################################################################
#                                 File Paths                                  #
###############################################################################  
class FilePaths():              
    def __init__(self, hyperp, run_options): 
        #=== Declaring File Name Components ===#
        self.autoencoder_type = 'VAE_'
        self.autoencoder_loss = 'maware'
        if run_options.data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if run_options.data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        self.N_Nodes = '_' + str(run_options.full_domain_dimensions) # Must begin with an underscore!
        if run_options.diagonal_posterior_covariance == 1:
            self.posterior_covariance_shape = 'diagpost_'
        if run_options.full_posterior_covariance == 1:
            self.posterior_covariance_shape = 'fullpost_'
        if run_options.fin_dimensions_2D == 1:
            fin_dimension = ''
        if run_options.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if run_options.prior_cov_length >= 1:
            run_options.prior_cov_length = int(run_options.prior_cov_length)
            prior_cov_length_string = str(run_options.prior_cov_length)
        else:
            prior_cov_length_string = str(run_options.prior_cov_length)
            prior_cov_length_string = 'pt' + prior_cov_length_string[2:]              
        if run_options.prior_elliptic_d_p >= 1:
            run_options.prior_elliptic_d_p = int(run_options.prior_elliptic_d_p)
            prior_elliptic_d_p_string = str(run_options.prior_elliptic_d_p)
        else:
            prior_elliptic_d_p_string = str(run_options.prior_elliptic_d_p)
            prior_elliptic_d_p_string = 'pt' + prior_elliptic_d_p_string[2:]
        if run_options.prior_elliptic_g_p >= 1:
            run_options.prior_elliptic_g_p = int(run_options.prior_elliptic_g_p)
            prior_elliptic_g_p_string = str(run_options.prior_elliptic_g_p)
        else:
            prior_elliptic_g_p_string = str(run_options.prior_elliptic_g_p)
            prior_elliptic_g_p_string = 'pt' + prior_elliptic_g_p_string[2:]
        
        #=== File Name ===#
        if run_options.prior_type_elliptic == 1:
            self.filename = self.autoencoder_type + self.posterior_covariance_shape + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_' + run_options.prior_type + '_dp%s_gp%s' %(prior_elliptic_d_p_string, prior_elliptic_g_p_string) + '_hl%d_tl%d_hn%d_%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        if run_options.prior_type_nonelliptic == 1:
            self.filename = self.autoencoder_type + self.posterior_covariance_shape + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_' + run_options.kern_type + '_cl%s' %(prior_cov_length_string) + '_hl%d_tl%d_hn%d_%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)

        #=== Save File Directory ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
  
        #=== Save File Path Observation Indices ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyperp.data_type + fin_dimension
    
        #=== Save File Path for Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test' + self.N_Nodes + fin_dimension + parameter_type
        if hyperp.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test' + self.N_Nodes + fin_dimension + parameter_type
        if hyperp.data_type == 'bndonly':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test_bnd' + self.N_Nodes + fin_dimension + parameter_type           
            
        #=== Loading Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test' + self.N_Nodes + fin_dimension + parameter_type
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test' + self.N_Nodes + fin_dimension + parameter_type
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'

        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)    

###############################################################################
#                                    Driver                                   #
###############################################################################
if __name__ == "__main__":
    
    #=== Hyperparameters and Run Options ===#    
    hyperp = Hyperparameters()
    run_options = RunOptions()
    
    if len(sys.argv) > 1:
        hyperp.data_type         = str(sys.argv[1])
        hyperp.num_hidden_layers = int(sys.argv[2])
        hyperp.truncation_layer  = int(sys.argv[3])
        hyperp.num_hidden_nodes  = int(sys.argv[4])
        hyperp.activation        = str(sys.argv[5])
        hyperp.batch_size        = int(sys.argv[6])
        hyperp.num_epochs        = int(sys.argv[7])
        
    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Plot and Save Matplotlib ===#
    fig_size = (5,5)
    plot_and_save_predictions(hyperp, run_options, file_paths, fig_size)
    plot_and_save_metrics(hyperp, run_options, file_paths, fig_size)
    
    #=== Plot and Save vtkfiles ===#
    #plot_and_save_predictions_vtkfiles(hyperp, run_options, file_paths)