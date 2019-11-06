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

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

###############################################################################
#                               Parameters                                    #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 3
    truncation_layer  = 2 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 614
    penalty           = 1
    num_training_data = 20
    batch_size        = 5
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
        self.num_testing_data = 20
        
        #=== File name ===#
        if hyper_p.penalty >= 1:
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        #=== Loading and saving data ===#
        if self.use_full_domain_data == 1:
            self.observation_indices_savefilepath = '../Datasets/' + 'thermal_fin_full_domain'
            self.parameter_train_savefilepath = '../Datasets/' + 'parameter_train_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../Datasets/' + 'state_train_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Datasets/' + 'parameter_test_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Datasets/' + 'state_test_%d' %(self.num_testing_data) 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.observation_indices_savefilepath = '../Datasets/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../Datasets/' + 'parameter_train_bnd_%d' %(hyper_p.num_training_data) 
            self.state_obs_train_savefilepath = '../Datasets/' + 'state_train_bnd_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Datasets/' + 'parameter_test_bnd_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Datasets/' + 'state_test_bnd_%d' %(self.num_testing_data)      
    
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test'
        self.figures_savefile_name_parameter_pred = self.figures_savefile_directory + '/' + 'parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_directory + '/' + 'state_pred'
        
        #=== Creating Directories ===#
        if not os.path.exists('../Datasets'):
            os.makedirs('../Datasets')
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
       
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
    parameter_and_state_obs_val_draw = parameter_and_state_obs_val.take(10)
    
    for batch_num, (parameter_val, state_obs_val) in parameter_and_state_obs_val_draw.enumerate():
        print(batch_num.numpy())
        parameter_pred = NN.decoder(state_obs_val)
        state_pred = NN.encoder(parameter_val)
    
    pdb.set_trace()
    