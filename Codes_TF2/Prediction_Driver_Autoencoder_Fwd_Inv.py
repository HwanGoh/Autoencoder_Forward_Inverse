#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""
import sys
sys.path.append('../')

from Utilities.predict_and_save import predict_and_save

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import sys

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    data_type         = 'bndonly'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 1
    num_training_data = 50000
    batch_size        = 1000
    num_epochs        = 500
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        self.num_testing_data = 200
        
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
        #=== Random Seed ===#
        self.random_seed = 1234

###############################################################################
#                                 File Name                                   #
###############################################################################         
        #=== Data Type Names ===#
        self.use_full_domain_data = 0
        self.use_bnd_data = 0
        self.use_bnd_data_only = 0
        if hyper_p.data_type == 'full':
            self.use_full_domain_data = 1
        if hyper_p.data_type == 'bnd':
            self.use_bnd_data = 1
        if hyper_p.data_type == 'bndonly':
            self.use_bnd_data_only = 1
        
        #=== Parameter and Observation Dimensions === #
        self.full_domain_dimensions = 1446 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        #=== File Name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
            parameter_type = ''
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
            parameter_type = '_vary'
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = self.dataset + '_' + hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
############################################################################### 
        #=== Loading Data ===#
        if self.use_full_domain_data == 1:
            self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_full_domain'
            self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(hyper_p.num_training_data) + parameter_type
            self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(hyper_p.num_training_data) + parameter_type
            self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) + parameter_type
            self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) + parameter_type
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_bnd_%d' %(hyper_p.num_training_data) + parameter_type
            self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_bnd_%d' %(hyper_p.num_training_data) + parameter_type
            self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_bnd_%d' %(self.num_testing_data) + parameter_type
            self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_bnd_%d' %(self.num_testing_data) + parameter_type          
        
        #=== Save File Name ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        
        #=== Save File Path for One Instance of Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_name + '_parameter_test'
        if hyper_p.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_name + '_state_test'
        if hyper_p.data_type == 'bndonly':
            self.savefile_name_state_test = self.NN_savefile_name + '_state_test_bnd'
            
        #=== Save File Path for Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'     
            
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
        hyper_p.num_training_data = int(sys.argv[6])
        hyper_p.batch_size        = int(sys.argv[7])
        hyper_p.num_epochs        = int(sys.argv[8])
        hyper_p.gpu               = str(sys.argv[9])
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    #=== Predict and Save ===#
    predict_and_save(hyper_p, run_options)
    
