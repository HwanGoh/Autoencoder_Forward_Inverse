#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""
from Utilities.predict_and_save import predict_and_save

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
import sys

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    data_type         = 'bnd'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 1
    batch_size        = 1000
    num_epochs        = 10
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_training_data = 200
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 1
        self.fin_dimensions_3D = 0
        
        #=== Random Seed ===#
        self.random_seed = 1234

###############################################################################
#                                 File Name                                   #
###############################################################################                
        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        
        #=== File name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = '_nine'
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if self.fin_dimensions_2D == 1:
            fin_dimension = ''
        if self.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, self.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
############################################################################### 
        #=== Loading and saving data ===#
        self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'obs_indices' + '_' + hyper_p.data_type + fin_dimension
        self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(self.num_training_data) + fin_dimension + parameter_type
        self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(self.num_training_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + fin_dimension + parameter_type 
        self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + fin_dimension + '_' + hyper_p.data_type + parameter_type
        
        #=== Save File Name ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        
        #=== Save File Path for One Instance of Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test' + fin_dimension
        if hyper_p.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test' + fin_dimension
        if hyper_p.data_type == 'bnd':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test_bnd' + fin_dimension
         
        #=== Save File Path for Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred' + fin_dimension
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred' + fin_dimension     
            
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Set hyperparameters ===#
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.data_type         = str(sys.argv[1])
        hyper_p.num_hidden_layers = int(sys.argv[2])
        hyper_p.truncation_layer  = int(sys.argv[3])
        hyper_p.num_hidden_nodes  = int(sys.argv[4])
        hyper_p.penalty           = float(sys.argv[5])
        hyper_p.batch_size        = int(sys.argv[6])
        hyper_p.num_epochs        = int(sys.argv[7])
        hyper_p.gpu               = str(sys.argv[8])
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    #=== Predict and Save ===#
    predict_and_save(hyper_p, run_options)
    
