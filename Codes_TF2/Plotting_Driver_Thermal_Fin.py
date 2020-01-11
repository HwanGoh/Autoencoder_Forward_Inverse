#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""
import sys
import os
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
    activation        = 'relu'
    penalty           = 10
    penalty_aug       = 1
    penalty_pr        = 0.0
    batch_size        = 1000
    num_epochs        = 1000
    
class RunOptions:
    def __init__(self): 
        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 1
        self.use_reverse_autoencoder = 0
        
        #=== Autoencoder Loss ===#
        self.use_model_aware = 1
        self.use_model_augmented = 0
        self.use_model_induced = 0
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 0
        self.data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_data_train = 10000
        self.num_data_test = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Prior Properties ===#
        if self.fin_dimensions_2D == 1:
            self.kern_type = 'sq_exp'
            self.prior_cov_length = 0.8
            self.prior_mean = 0.0
        if self.fin_dimensions_3D == 1:    
            self.kern_type = 'sq_exp'
            self.prior_cov_length = 0.8
            self.prior_mean = 0.0
        
        #=== Random Seed ===#
        self.random_seed = 1234

        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 4658 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 5047 
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
        if run_options.use_standard_autoencoder == 1:
            self.autoencoder_type = ''
        if run_options.use_reverse_autoencoder == 1:
            self.autoencoder_type = 'rev_'
        if run_options.use_model_aware == 1:
            self.autoencoder_loss = 'maware'
        if run_options.use_model_augmented == 1:
            self.autoencoder_loss = 'maug'
        if run_options.use_model_induced == 1:
            self.autoencoder_loss = 'mind'
        if run_options.data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if run_options.data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        self.N_Nodes = '_' + str(run_options.full_domain_dimensions) # Must begin with an underscore!
        if run_options.fin_dimensions_2D == 1 and run_options.full_domain_dimensions == 1446:
            self.N_Nodes = ''
        if run_options.fin_dimensions_3D == 1 and run_options.full_domain_dimensions == 4090:
            self.N_Nodes = ''
        if run_options.fin_dimensions_2D == 1:
            fin_dimension = ''
        if run_options.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyperp.penalty >= 1:
            hyperp.penalty = int(hyperp.penalty)
            penalty_string = str(hyperp.penalty)
        else:
            penalty_string = str(hyperp.penalty)
            penalty_string = 'pt' + penalty_string[2:]
        if hyperp.penalty_aug >= 1:
            hyperp.penalty_aug = int(hyperp.penalty_aug)
            penalty_string_aug = str(hyperp.penalty_aug)
        else:
            penalty_string_aug = str(hyperp.penalty_aug)
            penalty_string_aug = 'pt' + penalty_string_aug[2:]  
        if hyperp.penalty_pr >= 1:
            hyperp.penalty_pr = int(hyperp.penalty_pr)
            penalty_pr_string = str(hyperp.penalty_pr)
        else:
            penalty_pr_string = str(hyperp.penalty_pr)
            penalty_pr_string = 'pt' + penalty_pr_string[2:]
 
        #=== File Name ===#
        if run_options.use_model_aware == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_string, penalty_pr_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        if run_options.use_model_augmented == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_string, penalty_pr_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        if run_options.use_model_induced == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_paug%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_string, penalty_string_aug, penalty_pr_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        
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
        hyperp.penalty           = float(sys.argv[6])
        hyperp.penalty_aug       = float(sys.argv[7])
        hyperp.penalty_pr        = float(sys.argv[8])
        hyperp.batch_size        = int(sys.argv[9])
        hyperp.num_epochs        = int(sys.argv[10])

    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Plot and Save Matplotlib ===#
    #fig_size = (5,5)
    #plot_and_save_predictions(hyperp, run_options, file_paths, fig_size)
    #plot_and_save_metrics(hyperp, run_options, file_paths, fig_size)
    
    #=== Plot and Save vtkfiles ===#
    plot_and_save_predictions_vtkfiles(hyperp, run_options, file_paths)
    
    
    
