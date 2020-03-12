#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""
import sys
import os

from Utilities.plot_and_save_predictions_paraview import plot_and_save_predictions_paraview

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
    penalty_encoder   = 0.01
    penalty_decoder   = 0.01
    penalty_aug       = 0
    penalty_prior     = 0.0
    batch_size        = 1000
    num_epochs        = 1000
    
class RunOptions:
    def __init__(self): 
        #=== Autoencoder Type ===#
        self.use_standard_autoencoder = 0
        self.use_reverse_autoencoder = 1
        
        #=== Autoencoder Loss ===#
        self.use_model_aware = 1
        self.use_model_augmented = 0
        self.use_model_induced = 0
        
        #=== Data Set ===#
        self.data_thermal_fin_nine = 1
        self.data_thermal_fin_vary = 0
        
        #=== Data Set Size ===#
        self.num_data_train = 50000
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
        if hyperp.penalty_encoder >= 1:
            hyperp.penalty_encoder = int(hyperp.penalty_encoder)
            penalty_encoder_string = str(hyperp.penalty_encoder)
        else:
            penalty_encoder_string = str(hyperp.penalty_encoder)
            penalty_encoder_string = 'pt' + penalty_encoder_string[2:]
        if hyperp.penalty_decoder >= 1:
            hyperp.penalty_decoder = int(hyperp.penalty_decoder)
            penalty_decoder_string = str(hyperp.penalty_decoder)
        else:
            penalty_decoder_string = str(hyperp.penalty_decoder)
            penalty_decoder_string = 'pt' + penalty_decoder_string[2:]
        if hyperp.penalty_aug >= 1:
            hyperp.penalty_aug = int(hyperp.penalty_aug)
            penalty_string_aug = str(hyperp.penalty_aug)
        else:
            penalty_string_aug = str(hyperp.penalty_aug)
            penalty_string_aug = 'pt' + penalty_string_aug[2:]  
        if hyperp.penalty_prior >= 1:
            hyperp.penalty_proor = int(hyperp.penalty_prior)
            penalty_prior_string = str(hyperp.penalty_prior)
        else:
            penalty_prior_string = str(hyperp.penalty_prior)
            penalty_prior_string = 'pt' + penalty_prior_string[2:]
 
        #=== File Name ===#
        if run_options.use_model_aware == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_en%s_de%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_encoder_string, penalty_decoder_string, penalty_prior_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        if run_options.use_model_augmented == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_encoder_string, penalty_prior_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
        if run_options.use_model_induced == 1:
            self.filename = self.autoencoder_type + self.autoencoder_loss + '_' + self.dataset + self.N_Nodes + '_' + hyperp.data_type + fin_dimension + '_hl%d_tl%d_hn%d_%s_p%s_paug%s_pr%s_d%d_b%d_e%d' %(hyperp.num_hidden_layers, hyperp.truncation_layer, hyperp.num_hidden_nodes, hyperp.activation, penalty_encoder_string, penalty_string_aug, penalty_prior_string, run_options.num_data_train, hyperp.batch_size, hyperp.num_epochs)
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '/home/hwan/Documents/Github_Codes/Autoencoder_Forward_Inverse/Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test' + fin_dimension + parameter_type
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test' + fin_dimension + parameter_type
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
        hyperp.penalty_encoder   = float(sys.argv[6])
        hyperp.penalty_decoder   = float(sys.argv[7])
        hyperp.penalty_aug       = float(sys.argv[8])
        hyperp.penalty_prior     = float(sys.argv[9])
        hyperp.batch_size        = int(sys.argv[10])
        hyperp.num_epochs        = int(sys.argv[11])

    #=== File Names ===#
    file_paths = FilePaths(hyperp, run_options)
    
    #=== Plot and Save Paraview ===#
    if run_options.data_thermal_fin_nine == 1:
        cbar_RGB_parameter_test = [0.5035549, 0.231373, 0.298039, 0.752941, 1.3869196499999998, 0.865003, 0.865003, 0.865003, 2.2702843999999995, 0.705882, 0.0156863, 0.14902]
        cbar_RGB_state_test = [0.0018624861168711026, 0.231373, 0.298039, 0.752941, 0.7141493372496077, 0.865003, 0.865003, 0.865003, 1.4264361883823442, 0.705882, 0.0156863, 0.14902]
    if run_options.data_thermal_fin_vary == 1:
        cbar_RGB_parameter_test = [0.30348012, 0.231373, 0.298039, 0.752941, 1.88775191, 0.865003, 0.865003, 0.865003, 3.4720237000000003, 0.705882, 0.0156863, 0.14902]
        cbar_RGB_state_test = [0.004351256241582283, 0.231373, 0.298039, 0.752941, 0.5831443090996347, 0.865003, 0.865003, 0.865003, 1.1619373619576872, 0.705882, 0.0156863, 0.14902]
    
    plot_and_save_predictions_paraview(hyperp, file_paths, cbar_RGB_parameter_test, cbar_RGB_state_test)
    
    
    
