#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""

import sys
sys.path.append('../')

import pandas as pd

from Utilities.get_thermal_fin_data import load_thermal_fin_data
from Utilities.NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import sys

###############################################################################
#                               Parameters                                    #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 3
    truncation_layer  = 2 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 200
    penalty           = 1
    num_training_data = 50000
    batch_size        = 1000
    num_epochs        = 100
    gpu               = '2'
    
class RunOptions:
    def __init__(self, hyper_p):
        #=== Random Seed ===#
        self.random_seed = 1234
        
        #=== Data type ===#
        self.use_full_domain_data = 0
        self.use_bnd_data = 0
        self.use_bnd_data_only = 0
        if hyper_p.data_type == 'full':
            self.use_full_domain_data = 1
        if hyper_p.data_type == 'bnd':
            self.use_bnd_data = 1
        if hyper_p.data_type == 'bndonly':
            self.use_bnd_data_only = 1
        
        #=== Observation Dimensions ===#
        self.full_domain_dimensions = 1446 
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        #=== Number of Testing Data ===#
        self.num_testing_data = 2000
        
        #=== File name ===#
        if hyper_p.penalty >= 1:
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Loading and saving data ===#
        if self.use_full_domain_data == 1:
            self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_full_domain'
            self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_%d' %(self.num_testing_data) 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.observation_indices_savefilepath = '../../Datasets/Thermal_Fin/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_train_bnd_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_train_bnd_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'parameter_test_bnd_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../../Datasets/Thermal_Fin/' + 'state_test_bnd_%d' %(self.num_testing_data)      
    
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.figures_savefile_directory = '../Figures/' + self.filename
               
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Set hyperparameters ===#
    hyper_p = HyperParameters()
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    #=== Load observation indices ===#  
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(run_options.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy()    
    
    #=== Load testing data ===# 
    obs_indices, parameter_and_state_obs_train, parameter_and_state_obs_test, parameter_and_state_obs_val, data_input_shape, parameter_dimension, num_batches_train, num_batches_val = load_thermal_fin_data(run_options, hyper_p.num_training_data, hyper_p.batch_size, run_options.random_seed) 

    ####################################
    #   Import Trained Neural Network  #
    ####################################        
    #=== Neural Network ===#
    NN = AutoencoderFwdInv(hyper_p, run_options, data_input_shape[0], run_options.full_domain_dimensions, obs_indices, run_options.NN_savefile_name, construct_flag = 1)
    NN.load_weights(run_options.NN_savefile_name)
                
    #######################
    #   Form Predictions  #
    #######################      
    parameter_and_state_obs_val_draw = parameter_and_state_obs_val.take(1)
    
    for batch_num, (parameter_test, state_obs_test) in parameter_and_state_obs_val_draw.enumerate():
        parameter_pred_batch = NN.decoder(state_obs_test)
        state_pred_batch = NN.encoder(parameter_test)
    
    parameter_test = parameter_test[0,:].numpy()
    parameter_pred = parameter_pred_batch[0,:].numpy()
    if run_options.use_full_domain_data == 1: # No state prediction if the truncation layer only consists of the observations
        state_test = state_obs_test[0,:].numpy()
        state_pred = state_pred_batch[0,:].numpy()
    
    #######################
    #   Save Predictions  #
    #######################  
    df_parameter_test = pd.DataFrame({'parameter_test': parameter_test})
    df_parameter_test.to_csv(run_options.NN_savefile_name + '_parameter_test' + '.csv', index=False)  
    df_parameter_pred = pd.DataFrame({'parameter_pred': parameter_pred})
    df_parameter_pred.to_csv(run_options.NN_savefile_name + '_parameter_pred' + '.csv', index=False)  
    if run_options.use_full_domain_data == 1: # No state prediction if the truncation layer only consists of the observations
        df_state_test = pd.DataFrame({'state_test': state_test})
        df_state_test.to_csv(run_options.NN_savefile_name + '_state_test' + '.csv', index=False)  
        df_state_pred = pd.DataFrame({'state_pred': state_pred})
        df_state_pred.to_csv(run_options.NN_savefile_name + '_state_pred' + '.csv', index=False)  

    print('\nPredictions Saved to ' + run_options.NN_savefile_name)
        
    